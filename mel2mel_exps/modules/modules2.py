import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm
import torch.nn.functional as F

from torch import distributed

from constants import LRELU_SLOPE


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        # TODO CHECK SHAPES
        # TODO add mask, maskout paddings, dont count paddings in cluster_size and ect
        flatten = input.reshape(-1, self.dim)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            distributed.all_reduce(embed_onehot_sum)
            distributed.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, n_blocks=2):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [nn.Sequential(weight_norm(Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)),
                                         nn.ReLU(),
                                         weight_norm(Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)),
                                         nn.ReLU()) for _ in range(n_blocks)])

    def forward(self, x):
        for res_block in self.res_blocks:
            x = res_block(x) + x
        return x


class VQVAEModule(nn.Module):
    def __init__(self, n_mels, channel_size, resblock_kernel_sizes, embed_dim, n_embed):
        super().__init__()
        self.conv_pre = weight_norm(Conv1d(n_mels, channel_size, 3, 1, padding=1))
        self.conv_out = weight_norm(Conv1d(embed_dim // 2, n_mels, 5, 1, padding=2))
        self.resblocks_encoder = nn.ModuleList()
        self.resblocks_decoder = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.n_kernels = len(resblock_kernel_sizes)
        self.lrelu_slope = LRELU_SLOPE
        for i in range(3):
            resblock_list = nn.ModuleList()
            current_channel_size = channel_size * 2 ** i
            for kernel_size in resblock_kernel_sizes:
                resblock_list.append(ResBlock(current_channel_size, kernel_size))
            self.resblocks_encoder.append(resblock_list)

            self.downs.append(weight_norm(Conv1d(current_channel_size, 2 * current_channel_size, 3, 2, 1)))

        for i in range(3):
            if i == 0:
                self.ups.append(weight_norm(ConvTranspose1d(embed_dim, embed_dim, 4, 2, padding=1)))
            elif i == 1:
                self.ups.append(weight_norm(ConvTranspose1d(2 * embed_dim, embed_dim, 4, 2, padding=1)))
            else:
                self.ups.append(weight_norm(ConvTranspose1d(embed_dim, embed_dim // 2, 4, 2, padding=1)))

            resblock_list = nn.ModuleList()
            for kernel_size in resblock_kernel_sizes:
                if i == 2:
                    resblock_list.append(ResBlock(embed_dim // 2, kernel_size))
                else:
                    resblock_list.append(ResBlock(embed_dim, kernel_size))
            self.resblocks_decoder.append(resblock_list)

        self.quant_conv_top = weight_norm(nn.Conv1d(channel_size * 2 ** 3, embed_dim, 1))
        self.quant_conv_bottom = weight_norm(nn.Conv1d(channel_size * 2 ** 2 + embed_dim, embed_dim, 1))
        self.quantize_upsample = weight_norm(ConvTranspose1d(embed_dim, embed_dim, 4, 2, padding=1))

        self.quantize_bottom = Quantize(embed_dim, n_embed)
        self.quantize_top = Quantize(embed_dim, n_embed)


    def forward(self, x):
        x = self.conv_pre(x)
        for i, (resblocks, down) in enumerate(zip(self.resblocks_encoder, self.downs)):
            x = F.leaky_relu(x, self.lrelu_slope)
            xs = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
            for resblock in resblocks:
                xs += resblock(x)
            x = xs / self.n_kernels

            x = F.leaky_relu(x, self.lrelu_slope) # TODO no in hifi G
            x = down(x)

            if i == 1:
                encoded_bottom = x

        encoded_top = F.leaky_relu(x, self.lrelu_slope)
        encoded_top = self.quant_conv_top(encoded_top).permute(0, 2, 1)
        quant_top, diff_top, id_top = self.quantize_top(encoded_top)
        quant_top = quant_top.permute(0, 2, 1)

        x = self.ups[0](quant_top)  # TODO maybe better with relu?
        xs = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        for resblock in self.resblocks_decoder[0]:
            xs += resblock(x)
        x = xs / self.n_kernels

        x = torch.cat([encoded_bottom, x], 1)
        quant_bottom = self.quant_conv_bottom(x).permute(0, 2, 1)
        quant_bottom, diff_bottom, id_bottom = self.quantize_bottom(quant_bottom)
        quant_bottom = quant_bottom.permute(0, 2, 1)

        x = torch.cat([quant_bottom, self.quantize_upsample(quant_top)], 1)

        for up, resblocks in zip(self.ups[1:], self.resblocks_decoder[1:]):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = up(x)

            xs = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
            for resblock in resblocks:
                xs += resblock(x)
            x = xs / self.n_kernels

        x = F.leaky_relu(x, self.lrelu_slope)
        x = self.conv_out(x)
        return x, diff_bottom + diff_top









