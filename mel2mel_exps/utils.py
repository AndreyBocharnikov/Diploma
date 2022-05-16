import numpy as np
import torch
import torch.nn as nn
import torchaudio
from hydra.utils import instantiate

from constants import PAD_MEL_VALUE, MIN_MEL_VALUE


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class ComputeMel(nn.Module):
    def __init__(self, sample_rate, n_fft, hop_length, n_mels, power=1, center=False,
                 f_min=0, f_max=None, pad_value=PAD_MEL_VALUE):
        super().__init__()
        self.compute_mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft,
                                                                hop_length=hop_length, n_mels=n_mels,
                                                                power=power, center=center,
                                                                f_min=f_min, f_max=f_max,
                                                                norm='slaney', mel_scale='slaney')
        self.hop_length = hop_length
        self.padding = n_fft - hop_length
        self.pad_value = pad_value

    def forward(self, wav, wav_lens=None):
        padded_wav = torch.nn.functional.pad(wav,
                                             (self.padding // 2, self.padding // 2),
                                             mode='reflect')
        mel = log_mel(self.compute_mel(padded_wav))

        if wav_lens is None:
            return mel

        mel_lens = wav_lens // self.hop_length

        max_len = mel.size(-1)
        mask = torch.arange(max_len).to(mel.device)
        mask = mask.expand(mel.size(0), max_len) >= mel_lens.unsqueeze(1)
        mel = mel.masked_fill(mask.unsqueeze(1).type(torch.bool).to(device=mel.device), self.pad_value)

        return mel


def log_mel(mel):
    return torch.log(torch.clamp(mel, min=MIN_MEL_VALUE))  # torch.finfo(mel.dtype).eps


def pad_mel(mels, max_mel_len=None):
    # mels - list[n_mels, T]
    if max_mel_len is None:
        max_mel_len = max([mel.shape[1] for mel in mels])
    padded_mels = np.zeros((len(mels), mels[0].shape[0], max_mel_len)) + PAD_MEL_VALUE
    for i, mel in enumerate(mels):
        padded_mels[i, :, :mel.shape[1]] = mel
    return torch.tensor(padded_mels, dtype=torch.float)


def get_vocoder_generator(state_dict: dict):
    generator = {}
    for key, value in state_dict.items():
        if "generator" in key:
            generator[key[10:]] = value
    return generator


def load_vocoder(cfg):
    vocoder = instantiate(cfg.vocoder)
    checkpoint = torch.load(cfg.pretrained_vocoder_path, map_location=torch.device('cpu'))
    generator_state_dict = get_vocoder_generator(checkpoint['state_dict'])
    vocoder.load_state_dict(generator_state_dict)
    vocoder.remove_weight_norm()
    vocoder.eval()
    return vocoder


def log_wavs(vocoder, logger, global_step, sr, mels, mel_lens, n, handle):
    with torch.no_grad():
        for i in range(n):
            wav = vocoder(mels[i: i + 1, :, :mel_lens[i]])
            logger.experiment.add_audio(handle + f" {str(i)}", wav.detach().cpu().numpy(), global_step=global_step,
                                        sample_rate=sr)
