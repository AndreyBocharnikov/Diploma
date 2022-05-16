import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torchaudio
import tqdm


def compute_non_silente_dur_vctk():
    path = '/Users/a.bocharnikov/PycharmProjects/avc/vctk/for_avc'
    wavs = list(Path(path).glob('*.flac'))
    vad = torchaudio.transforms.Vad(48000)
    print(len(wavs))
    for wav_path in tqdm.tqdm(wavs):
        wav, sr = torchaudio.load(wav_path)
        wav = vad(wav)

        wav = torch.from_numpy(np.array(wav.numpy().tolist()[0][::-1])).unsqueeze(dim=0)

        wav = vad(wav)
        dur = wav.shape[1] / sr
        name = str(wav_path).split('/')[-1].split('.')[0] #
        if (dur > 2.5) and '243' not in name and '226' not in name and '227' not in name and '225' not in name:
            print(dur, wav_path)


def resample():
    path = '/Users/a.bocharnikov/PycharmProjects/avc/vctk/for_avc'
    p = list(Path(path).glob('**/*.flac'))
    for path in tqdm.tqdm(p):
        path = str(path)
        wav, sr = torchaudio.load(path)
        assert sr == 48000, path
        name = path.split('/')[-1].split('.')[0]
        basepath = '/'.join(path.split('/')[:-1])
        subprocess.call(["sox", path, "-r", "22050", "-c", "1", os.path.join(basepath, name + ".wav")])


def make_val_manifest_right_order():
    target_sample_paths = '/root/storage/TTS/avc/val_samples'
    long_samples = ['p243_098_mic1', 'p243_096_mic1', 'p227_087_mic1',  'p227_080_mic1',
                    'p226_072_mic1', 'p226_071_mic1', 'p225_037_mic1', 'p225_030_mic1',
                    'p233_044_mic1', 'p226_071_mic1', 'p267_068_mic1', 'p233_044_mic1',
                    'p259_124_mic1',  'p232_123_mic1', 'p228_117_mic1', 'p239_134_mic1']
    all_samples_path = '/Users/a.bocharnikov/PycharmProjects/AVC/val_manifest.txt'
    all_samples = [sample.strip() for sample in open(all_samples_path)]
    right_order = []
    idx = 0
    long_id = 0
    for i in range(len(all_samples)):
        if idx % 4 == 0 and long_id < len(long_samples):
            name = long_samples[long_id]
            right_order.append(os.path.join(target_sample_paths, name + '.wav'))
            long_id += 1
            idx += 1
        name = all_samples[i].split('/')[-1].split('.')[0]
        if name in long_samples:
            continue
        right_order.append(os.path.join(target_sample_paths, name + '.wav'))
        idx += 1

    target_path = '/Users/a.bocharnikov/PycharmProjects/AVC/val_manifest_right_order.txt'
    with open(target_path, 'w') as f:
        f.write('\n'.join(right_order))


def create_hifi_manifest():
    def split(source_path, target_path):
        with open(source_path) as f, open(target_path, 'w') as t:
            samples = f.readlines()
            for sample in tqdm.tqdm(samples):
                sample = sample.strip()
                metainfo = torchaudio.info(sample)
                dur = metainfo.num_frames // metainfo.sample_rate
                entry = {
                    "audio_filepath": sample,
                    "duration": dur,
                    "text": "kek"
                }
                t.write(json.dumps(entry) + '\n')

    source_train_path = "/root/storage/TTS/avc/manifests/avc_train.txt"
    target_train_path = "/root/storage/TTS/avc/manifests/avc_train_hifi.txt"

    source_val_path = '/root/storage/TTS/avc/manifests/val_manifest_right_order.txt'
    target_val_path = '/root/storage/TTS/avc/manifests/val_manifest_right_order_hifi.txt'

    split(source_train_path, target_train_path)
    split(source_val_path, target_val_path)


if __name__ == "__main__":
    # make_val_manifest_right_order()
    # resample()
    # compute_non_silente_dur_vctk()
    # create_hifi_manifest()
