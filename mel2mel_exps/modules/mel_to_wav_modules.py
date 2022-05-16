import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, Conv2d, AvgPool1d
from torch.nn.utils import weight_norm, spectral_norm
import torch.nn.functional as F

from torch import distributed

from constants import LRELU_SLOPE
from utils import get_padding, init_weights


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


class ResBlock1(torch.nn.Module):
    __constants__ = ['lrelu_slope']

    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.lrelu_slope = LRELU_SLOPE
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.lrelu_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.lrelu_slope)
            xt = c2(xt)
            x = xt + x
        return x


class Generator(nn.Module):
    __constants__ = ['lrelu_slope', 'num_kernels', 'num_upsamples']

    def __init__(
        self,
        resblock,
        upsample_rates,
        upsample_kernel_sizes,
        upsample_initial_channel,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        initial_input_size=80,
        apply_weight_init_conv_pre=False,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(Conv1d(initial_input_size, upsample_initial_channel, 7, 1, padding=3))
        self.lrelu_slope = LRELU_SLOPE
        resblock = ResBlock1

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            resblock_list = nn.ModuleList()
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                resblock_list.append(resblock(ch, k, d))
            self.resblocks.append(resblock_list)

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        if apply_weight_init_conv_pre:
            self.conv_pre.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for upsample_layer, resblock_group in zip(self.ups, self.resblocks):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = upsample_layer(x)
            xs = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
            for resblock in resblock_group:
                xs += resblock(x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x


class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False, debug=False):
        super().__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        conv_ch = [32, 128, 512, 1024] if not debug else [8, 12, 32, 64]
        self.convs = nn.ModuleList(
            [
                norm_f(Conv2d(1, conv_ch[0], (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(conv_ch[0], conv_ch[1], (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(conv_ch[1], conv_ch[2], (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(conv_ch[2], conv_ch[3], (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(conv_ch[3], conv_ch[3], (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(conv_ch[3], 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(2, debug=debug),
                DiscriminatorP(3, debug=debug),
                DiscriminatorP(5, debug=debug),
                DiscriminatorP(7, debug=debug),
                DiscriminatorP(11, debug=debug),
            ]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False, debug=False):
        super().__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        conv_ch = [128, 256, 512, 1024] if not debug else [16, 32, 32, 64]
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, conv_ch[0], 15, 1, padding=7)),
                norm_f(Conv1d(conv_ch[0], conv_ch[0], 41, 2, groups=4, padding=20)),
                norm_f(Conv1d(conv_ch[0], conv_ch[1], 41, 2, groups=16, padding=20)),
                norm_f(Conv1d(conv_ch[1], conv_ch[2], 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(conv_ch[2], conv_ch[3], 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(conv_ch[3], conv_ch[3], 41, 1, groups=16, padding=20)),
                norm_f(Conv1d(conv_ch[3], conv_ch[3], 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(conv_ch[3], 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(use_spectral_norm=True, debug=debug),
                DiscriminatorS(debug=debug),
                DiscriminatorS(debug=debug),
            ]
        )
        self.meanpools = nn.ModuleList([AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


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
    def __init__(self, n_mels, channel_size, resblock_kernel_sizes, n_resblocks, embed_dim, n_embed):
        super().__init__()
        self.conv_pre = weight_norm(Conv1d(n_mels, channel_size, 3, 1, padding=1))
        self.conv_out = weight_norm(Conv1d(embed_dim, n_mels, 1))
        self.resblocks_encoder = nn.ModuleList()
        self.between_resblocks_encoder = nn.ModuleList()
        self.resblocks_decoder = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.n_kernels = len(resblock_kernel_sizes)
        self.lrelu_slope = LRELU_SLOPE
        for i in range(2):
            resblock_list = nn.ModuleList()
            current_channel_size = channel_size * 2 ** i
            for kernel_size in resblock_kernel_sizes:
                resblock_list.append(ResBlock(current_channel_size, kernel_size, n_resblocks))
            self.resblocks_encoder.append(resblock_list)

            if i < 2 - 1:
                self.between_resblocks_encoder.append(weight_norm(Conv1d(current_channel_size, current_channel_size * 2, 1)))

        self.quant_conv_top = weight_norm(nn.Conv1d(channel_size * 2, embed_dim, 1))
        self.quantize_top = Quantize(embed_dim, n_embed)

    def forward(self, x):
        x = self.conv_pre(x)
        for i, resblocks in enumerate(self.resblocks_encoder):
            x = F.leaky_relu(x, self.lrelu_slope)
            xs = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
            for resblock in resblocks:
                xs += resblock(x)
            x = xs / self.n_kernels

            x = F.leaky_relu(x, self.lrelu_slope)
            if i < len(self.resblocks_encoder) - 1:
                x = self.between_resblocks_encoder[i](x)

        encoded_top = x
        encoded_top = self.quant_conv_top(encoded_top).permute(0, 2, 1)
        quant_top, diff_top, id_top = self.quantize_top(encoded_top)
        x = quant_top.permute(0, 2, 1)

        return x, diff_top









