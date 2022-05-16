import os

import pytorch_lightning as pl
import torch
import torchaudio
from hydra.utils import instantiate
from torch.utils.data import Dataset, DataLoader

import utils
from utils import pad_mel


class ParallelAlignedDataset(Dataset):
    def __init__(self, manifest, target_speaker_path, n_frames, mel_cfg):
        self.target_speaker_path = target_speaker_path
        self.wav_paths = []
        for sample in open(manifest):
            sample = sample.strip()
            metainfo = torchaudio.info(sample)
            dur = metainfo.num_frames // mel_cfg.hop_length
            if dur > n_frames:
                self.wav_paths.append(sample)
        # self.wav_paths = [sample.strip() for sample in open(manifest)]
        self.n_frames = n_frames
        print(mel_cfg)
        self.hop = mel_cfg.hop_length
        self.sample_rate = mel_cfg.sample_rate
        self.to_mel = utils.ComputeMel(**mel_cfg)

    def __getitem__(self, item):
        source_wav_path = self.wav_paths[item]
        sample_name = source_wav_path.split('/')[-1].split('.')[0]
        target_wav_path = os.path.join(self.target_speaker_path, sample_name + '.wav')

        if self.n_frames == -1:
            source_wav, _ = torchaudio.load(source_wav_path)
            # target_wav, _ = torchaudio.load(target_wav_path)

            source_mel = self.to_mel(source_wav).squeeze(dim=0)
            # target_mel = self.to_mel(target_wav)
            # return source_mel, target_mel
            mel_lens = source_mel.shape[1]
            return source_wav, source_mel, mel_lens

        meta_source = torchaudio.info(source_wav_path)
        meta_target = torchaudio.info(target_wav_path)

        # assert meta_source.num_frames // self.hop == meta_target.num_frames // self.hop, \
        #     str(source_wav_path) + ' ' + str(target_wav_path)
        assert self.sample_rate == meta_source.sample_rate
        assert self.sample_rate == meta_target.sample_rate

        total_frames = min(meta_target.num_frames // self.hop, meta_source.num_frames // self.hop)
        assert self.n_frames < total_frames, source_wav_path
        start_frame = torch.randint(total_frames - self.n_frames, [1]).item()
        start_sample = start_frame * self.hop
        num_samples = self.n_frames * self.hop

        source_wav, _ = torchaudio.load(source_wav_path, frame_offset=start_sample, num_frames=num_samples)
        target_wav, _ = torchaudio.load(target_wav_path, frame_offset=start_sample, num_frames=num_samples)

        source_mel = self.to_mel(source_wav).squeeze(dim=0)
        target_mel = self.to_mel(target_wav).squeeze(dim=0)

        return source_wav, source_mel, target_wav, target_mel

    def __len__(self):
        return len(self.wav_paths)

    def collate_fn(self, batch):
        if self.n_frames != -1:
            return torch.utils.data.dataloader.default_collate(batch)

        source_wavs, source_mel, mel_lens = zip(*batch)
        source_padded_mels = pad_mel(source_mel)
        max_wav_len = max([source_wav.shape[1] for source_wav in source_wavs])
        padded_wavs = torch.zeros((len(source_wavs), max_wav_len))
        for i, source_wav in enumerate(source_wavs):
            padded_wavs[i, :source_wav.shape[1]] = source_wav[0]
        mel_lens = torch.LongTensor(mel_lens)
        return padded_wavs, source_padded_mels, mel_lens
        # source_mels, target_mels = zip(batch)
        # source_padded_mels = pad_mel(source_mels)
        # target_padded_mels = pad_mel(target_mels)
        # return source_padded_mels, target_padded_mels


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


if __name__ == "__main__":
    from pathlib import Path
    target_path = '/root/storage/TTS/avc/manifests/avc_train.txt'
    path = '/root/storage/TTS/avc/generated_wavs'
    p = Path(path)
    paths = list(p.glob('**/*.wav'))
    print(len(paths), paths[0])
    with open(target_path, 'w') as f:
        f.write('\n'.join(map(str, paths)))