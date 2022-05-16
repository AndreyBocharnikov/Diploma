import subprocess
from pathlib import Path
import tqdm
import torch
import numpy as np
import torchaudio
import os
from multiprocessing import Pool


def resample_subarray(subarray):
    target_path = "wav16_trimmed"

    resampler = torchaudio.transforms.Resample(48000, 16000)
    vad = torchaudio.transforms.Vad(16000)
    iter_over = tqdm.tqdm(subarray) if str(subarray[0]) == 'wav48_silence_trimmed/p304/p304_117_mic1.flac' else subarray
    for path in iter_over:
        path = str(path)
        wav, sr = torchaudio.load(path)
        assert sr == 48000, path

        resampled_wav = resampler(wav)

        resampled_wav = vad(resampled_wav)
        resampled_wav = torch.from_numpy(np.array(resampled_wav.numpy().tolist()[0][::-1])).unsqueeze(dim=0)
        vadded_wav = vad(resampled_wav)
        vadded_wav = torch.from_numpy(np.array(vadded_wav.numpy().tolist()[0][::-1])).unsqueeze(dim=0)

        name = path.split('/')[-1].split('.')[0]
        speaker = path.split('/')[-2]
        vadded_wad_dir_path = os.path.join(target_path, speaker)
        os.makedirs(vadded_wad_dir_path, exist_ok=True)
        vadded_wad_path = os.path.join(vadded_wad_dir_path, name + ".wav")

        # print(vadded_wad_path)
        torchaudio.save(vadded_wad_path, vadded_wav.float().cpu(), 16000, encoding="PCM_S", bits_per_sample=16)


def resample():
    path = 'wav48_silence_trimmed'
    p = list(Path(path).glob('**/*_mic1.flac'))
    subarrays = np.array_split(p, 64)
    with Pool(64) as p:
        p.map(resample_subarray, subarrays)


def fix():
    path = '/root/storage/TTS/vctk/wav16_trimmed'
    p = list(Path(path).glob('**/*.wav'))
    for path in p:
        path = str(path)
        wav, sr = torchaudio.load(path)
        wav = torch.from_numpy(np.array(wav.numpy().tolist()[0][::-1])).unsqueeze(dim=0)
        torchaudio.save(path, wav.float().cpu(), 16000, encoding="PCM_S", bits_per_sample=16)


if __name__ == "__main__":
    # resample()
    fix()