import torch
import torch.nn as nn

from torch.nn import functional as F
from torch import distributed
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

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
    def __init__(self, in_channels, channels):
        super().__init__()
        self.net = nn.Sequential(nn.ReLU(),
                                 nn.Conv1d(in_channels, channels, 3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv1d(channels, in_channels, 1))

    def forward(self, x):
        return self.net(x) + x


class EncoderLayer(nn.Module):
    def __init__(self, convs, n_res_blocks, in_resblock_channels, resblock_channel_size):
        super().__init__()
        self.convs = convs
        self.resblocks = nn.Sequential(*[ResBlock(in_resblock_channels, resblock_channel_size)
                                         for _ in range(n_res_blocks)])

    def forward(self, x):
        x = self.convs(x)
        x = self.resblocks(x)
        return F.relu(x)


class DecoderLayer(nn.Module):
    def __init__(self, convs, in_channels, channels, n_res_blocks, resblock_channel_size):
        super().__init__()
        self.resblocks = nn.Sequential(nn.Conv1d(in_channels, channels, 3, padding=1),
                                       *[ResBlock(channels, resblock_channel_size)
                                         for _ in range(n_res_blocks)])
        self.convs = convs

    def forward(self, x):
        return self.convs(self.resblocks(x))


class VQVAEModule(nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 embed_dim, n_embed, n_res_blocks, resblock_channel_size):
        super().__init__()
        bottom_encoder_convs = nn.Sequential(nn.Conv1d(in_channels, hidden_channels // 2, 4, stride=2, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv1d(hidden_channels // 2, hidden_channels, 4, stride=2, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1))

        self.bottom_encoder = EncoderLayer(bottom_encoder_convs, n_res_blocks, hidden_channels, resblock_channel_size)

        top_encoder_convs = nn.Sequential(nn.Conv1d(hidden_channels, hidden_channels // 2, 4, stride=2, padding=1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv1d(hidden_channels // 2, hidden_channels, 3, padding=1))

        self.top_encoder = EncoderLayer(top_encoder_convs, n_res_blocks, hidden_channels, resblock_channel_size)

        top_decoder_convs = nn.ConvTranspose1d(hidden_channels, embed_dim, 4, stride=2, padding=1)
        self.top_decoder = DecoderLayer(top_decoder_convs, embed_dim, hidden_channels, n_res_blocks,
                                        resblock_channel_size)

        bottom_decoder_convs = nn.Sequential(nn.ConvTranspose1d(hidden_channels, hidden_channels // 2,
                                                                4, stride=2, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.ConvTranspose1d(hidden_channels // 2, in_channels,
                                                                4, stride=2, padding=1))
        self.bottom_decoder = DecoderLayer(bottom_decoder_convs, 2 * embed_dim, hidden_channels, n_res_blocks,
                                           resblock_channel_size)

        self.quantize_conv_bottom = nn.Conv1d(embed_dim + hidden_channels, embed_dim, 1)
        self.quantize_bottom = Quantize(embed_dim, n_embed)
        self.quantize_conv_top = nn.Conv1d(hidden_channels, embed_dim, 1)
        self.quantize_top = Quantize(embed_dim, n_embed)

        self.upsample_top = nn.ConvTranspose1d(embed_dim, embed_dim, 4, stride=2, padding=1)

    def forward(self, x):
        encoded_bottom = self.bottom_encoder(x)
        encoded_top = self.top_encoder(encoded_bottom)
        encoded_top = self.quantize_conv_top(encoded_top).permute(0, 2, 1)

        quant_top, diff_top, id_top = self.quantize_top(encoded_top)
        quant_top = quant_top.permute(0, 2, 1)

        decoded_top = self.top_decoder(quant_top)

        encoded_bottom = torch.cat([decoded_top, encoded_bottom], 1)
        quant_bottom = self.quantize_conv_bottom(encoded_bottom).permute(0, 2, 1)

        quant_bottom, diff_bottom, id_bottom = self.quantize_bottom(quant_bottom)
        quant_bottom = quant_bottom.permute(0, 2, 1)
        upsampled_top = self.upsample_top(quant_top)
        quant = torch.cat([upsampled_top, quant_bottom], 1)
        conversed = self.bottom_decoder(quant)
        # TODO resblock after upsampling like G of hifigan
        return conversed, diff_top + diff_bottom


class DiscriminatorSMel(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        conv_ch = [128, 256, 512, 1024]
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(1, conv_ch[0], 5, 2, padding=2)), # TODO check padding
                # norm_f(nn.Conv1d(conv_ch[0], conv_ch[0], 5, 1, padding=2)),
                norm_f(nn.Conv1d(conv_ch[0], conv_ch[1], 3, 1, padding=1)),
                # norm_f(nn.Conv1d(conv_ch[1], conv_ch[1], 3, 1, padding=1)),
                norm_f(nn.Conv1d(conv_ch[1], conv_ch[2], 3, 1, padding=1)),
                # norm_f(nn.Conv1d(conv_ch[2], conv_ch[2], 3, 1, padding=1)),
                norm_f(nn.Conv1d(conv_ch[2], conv_ch[3], 3, 1, padding=1)),
                norm_f(nn.Conv1d(conv_ch[3], conv_ch[3], 3, 1, padding=1)),
                norm_f(nn.Conv1d(conv_ch[3], conv_ch[3], 3, 1, padding=1)),
            ]
        )
        self.conv_post = norm_f(nn.Conv1d(conv_ch[3], 1, 3, 1, padding=1))

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


class MultiScaleDiscriminatorMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorSMel(use_spectral_norm=True),
                DiscriminatorSMel(),
                DiscriminatorSMel(),
            ]
        )
        self.meanpools = nn.ModuleList([nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)])

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


if __name__ == "__main__":
    in_channels = 80
    hidden_channels = 128
    embed_dim = 64
    n_embed = 512
    n_res_blocks = 2
    resblock_channel_size = 32
    model = VQVAEModule(in_channels, hidden_channels, embed_dim, n_embed, n_res_blocks, resblock_channel_size)

    mel = torch.rand((8, in_channels, 256))

    converted_mel, _ = model(mel)
    print(converted_mel.shape)
