import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from modules import Encoder, Quantize


class VQ_VAEModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.encoder_n_channels)
        self.conv_quantize = nn.Sequential(nn.LeakyReLU(0.1),
                                           nn.Conv1d(cfg.encoder_n_channels * 4, cfg.quantize.dim, 1))
        self.quantize = Quantize(cfg.quantize.dim, cfg.quantize.n_embed)

        self.emb_g = nn.Embedding(cfg.n_speakers, cfg.generator.gin_channels)
        speaker_embeds = np.load(cfg.speaker_embed_path)
        self.emb_g.weight = nn.Parameter(torch.from_numpy(speaker_embeds).float())

        self.generator = instantiate(cfg.generator)

    def forward(self, wavs, speaker_ids):
        encoded_wav = self.encoder(wavs)  # B, 1, T -> B, 32, T / 64 ; B, 1, T -> B, 512, T / 64
        encoded_wav = self.conv_quantize(encoded_wav).permute(0, 2, 1)
        vq_vae_embed, diff, _ = self.quantize(encoded_wav)
        vq_vae_embed = vq_vae_embed.permute(0, 2, 1)

        speaker_embed = self.emb_g(speaker_ids).unsqueeze(dim=-1)
        speaker_embed = speaker_embed.expand(speaker_embed.shape[0], speaker_embed.shape[1], vq_vae_embed.shape[2])

        generated_wavs = self.generator(vq_vae_embed, speaker_embed)
        return generated_wavs, diff
