import json
import os
import random

import pytorch_lightning as pl
import torch
import torchaudio
from hydra.utils import instantiate
from torch.utils.data import Dataset, DataLoader

import utils


def pad_wavs(wavs):
    max_wav_len = max(wav.shape[0] for wav in wavs)
    padded_wavs = torch.zeros((len(wavs), max_wav_len))
    for i, wav in enumerate(wavs):
        padded_wavs[i, :wav.shape[0]] = wavs[i]
    return padded_wavs


class VQVAEParallel(Dataset):
    def __init__(self, manifest, speaker_mapping, num_samples):
        self.wav_paths = []
        for sample in open(manifest):
            sample = sample.strip()
            meta_info = torchaudio.info(sample)
            if meta_info.num_frames > num_samples:
                self.wav_paths.append(sample)
        self.speaker_to_id_mapping = json.load(open(speaker_mapping)) # p376 -> 71
        speaker_embed_id_path = '/root/storage/TTS/vctk/speaker_mapping.txt'
        self.speaker_to_embed_mapping = json.load(open(speaker_embed_id_path)) # p376 -> 64
        # print(self.speaker_mapping)
        self.speakers = list(self.speaker_to_id_mapping.keys())
        # print(self.speakers)
        self.num_samples = num_samples
        self.val_speakers = ["p225", "p233", "p236", "p226", "p227", "p243", "p267", "p270"]
        # ["101", "84", ]
        # self.missing_alligned_samples = set()

    def __getitem__(self, item):
        source_path = self.wav_paths[item]
        if self.num_samples == -1:
            target_speaker_index = random.randint(0, len(self.val_speakers) - 1)
            target_speaker = self.val_speakers[target_speaker_index]
            target_speaker_id = self.speaker_to_id_mapping[target_speaker]
            target_speaker_embed_id = self.speaker_to_embed_mapping[target_speaker]

            source_speaker = source_path.split('/')[-2]
            target_speaker_path = source_path.replace(source_speaker, str(target_speaker_id))
            source_wav, s_sr = torchaudio.load(source_path)
            if not os.path.isfile(target_speaker_path):
                # if target_speaker_path not in self.missing_alligned_samples:
                #     print(target_speaker_path)
                #     self.missing_alligned_samples.add(target_speaker_path)
                target_speaker_path = source_path
            target_wav, t_sr = torchaudio.load(target_speaker_path)
            assert s_sr == t_sr
            return source_wav.squeeze(dim=0), target_wav.squeeze(dim=0), target_speaker_embed_id

        target_speaker_index = random.randint(0, len(self.speaker_to_id_mapping) - 1)
        target_speaker = self.speakers[target_speaker_index]
        target_speaker_id = self.speaker_to_id_mapping[target_speaker]
        target_speaker_id_embed = self.speaker_to_embed_mapping[target_speaker]

        tmp = source_path.split('/')
        source_speaker = tmp[-2]
        tmp[-2] = str(target_speaker_id)
        target_speaker_path = '/'.join(tmp)
        # target_speaker_path = source_path.replace(source_speaker, target_speaker)

        meta_source = torchaudio.info(source_path)
        meta_target = torchaudio.info(target_speaker_path)

        assert meta_source.num_frames == meta_target.num_frames

        start_frame = torch.randint(meta_source.num_frames - self.num_samples, [1]).item()
        source_wav, s_sr = torchaudio.load(source_path, frame_offset=start_frame, num_frames=self.num_samples)
        target_wav, t_sr = torchaudio.load(target_speaker_path, frame_offset=start_frame, num_frames=self.num_samples)
        assert s_sr == t_sr

        return source_wav.squeeze(dim=0), target_wav.squeeze(dim=0), target_speaker_id_embed

    def __len__(self):
        return len(self.wav_paths)

    def collate_fn(self, batch):
        if self.num_samples != -1:
            return torch.utils.data.dataloader.default_collate(batch)

        source_wavs, target_wavs, speakers = zip(*batch)
        padded_source_wavs = pad_wavs(source_wavs)
        padded_target_wavs = pad_wavs(target_wavs)
        return padded_source_wavs, padded_target_wavs, torch.LongTensor(speakers)


class VQVAEDataset(Dataset):
    def __init__(self, manifest, speaker_mapping, num_samples):
        self.wav_paths = []
        for sample in open(manifest):
            sample = sample.strip()
            meta_info = torchaudio.info(sample)
            if meta_info.num_frames > num_samples:
                self.wav_paths.append(sample)
        self.speaker_mapping = json.load(open(speaker_mapping))
        self.num_samples = num_samples

    def __getitem__(self, item):
        wav_path = self.wav_paths[item]
        speaker_name = wav_path.split('/')[-2]
        speaker_id = self.speaker_mapping[speaker_name]
        if self.num_samples > 0:
            meta_info = torchaudio.info(wav_path)
            start_frame = torch.randint(meta_info.num_frames - self.num_samples, [1]).item()
            wav, sr = torchaudio.load(wav_path, frame_offset=start_frame, num_frames=self.num_samples)
        else:
            wav, sr = torchaudio.load(wav_path)

        return wav.squeeze(dim=0), speaker_id

    def __len__(self):
        return len(self.wav_paths)

    def collate_fn(self, batch):
        if self.num_samples != -1:
            return torch.utils.data.dataloader.default_collate(batch)

        wavs, speakers = zip(*batch)
        padded_wavs = pad_wavs(wavs)
        return padded_wavs, torch.LongTensor(speakers)


class LightningDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.train_cfg = cfg.train_cfg
        self.val_cfg = cfg.val_cfg

    def setup(self, stage=None):
        self.train_dataset = instantiate(self.train_cfg.dataset)
        self.val_dataset = instantiate(self.val_cfg.dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          **self.train_cfg.dataloader)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          collate_fn=self.val_dataset.collate_fn,
                          **self.val_cfg.dataloader)


def make_manifests():
    from pathlib import Path
    path = Path('/root/storage/TTS/vctk/wav16_trimmed')
    paths = list(path.glob("**/*.wav"))
    print(len(paths))

    train_manifest_path = '/root/storage/TTS/vctk/manifests/train_manifest.txt'
    val_manifest_path = '/root/storage/TTS/vctk/manifests/val_manifest.txt'
    val_samples = ['001', '002', '003', '004', '005']

    with open(train_manifest_path, 'w') as f, open(val_manifest_path, 'w') as t:
        for path in paths:
            sample = str(path).split('/')[-1].split('_')[1]
            if sample in val_samples:
                t.write(str(path) + '\n')
            else:
                f.write(str(path) + '\n')


def make_speaker_mapping():
    from pathlib import Path
    path = Path('/root/storage/TTS/vctk/wav16_trimmed')
    speakers = list(path.glob('*'))
    speaker_mapping = dict()
    for i, speaker in enumerate(speakers):
        speaker = str(speaker).split('/')[-1]
        speaker_mapping[speaker] = i

    speaker_mapping_path = '/root/storage/TTS/vctk/speaker_mapping.txt'
    with open(speaker_mapping_path, 'w') as f:
        f.write(json.dumps(speaker_mapping))


if __name__ == "__main__":
    make_speaker_mapping()
