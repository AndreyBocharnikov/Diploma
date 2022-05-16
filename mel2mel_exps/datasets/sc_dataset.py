import json

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

import utils


class SCDatset(Dataset):
    def __init__(self, manifest, n_frames, mel_cfg, speaker_ids_path, val=False, speaker_mapping_path=None):
        self.val = val
        self.speaker_mapping = dict()
        for i, speaker in enumerate(open(speaker_ids_path).readlines()):
            self.speaker_mapping[speaker.strip()] = i
        if val:
            self.speaker_mapping2 = json.load(open(speaker_mapping_path))
        self.wav_paths = []
        for sample in open(manifest):
            sample = sample.strip()
            metainfo = torchaudio.info(sample)
            dur = metainfo.num_frames // mel_cfg.hop_length
            if dur > n_frames:
                self.wav_paths.append(sample)
        self.n_frames = n_frames
        self.hop = mel_cfg.hop_length
        self.to_mel = utils.ComputeMel(**mel_cfg)

    def __getitem__(self, item):
        wav_path = self.wav_paths[item]
        num_samples = self.n_frames * self.hop
        if self.val:
            wav, sr = torchaudio.load(wav_path)
            vad = torchaudio.transforms.Vad(22050)
            wav = vad(wav)
            wav = torch.from_numpy(np.array(wav.numpy().tolist()[0][::-1])).unsqueeze(dim=0)
            wav = vad(wav)
            start_sample = torch.randint(wav.shape[1] - num_samples, [1]).item()
            wav = wav[:, start_sample: start_sample + num_samples].float()
            # print(wav_path)
            speaker = wav_path.split('/')[-1].split('_')[0]
            # print(speaker)
            # print(self.speaker_mapping2)
            speaker = str(self.speaker_mapping2[speaker])
        else:
            metainfo = torchaudio.info(wav_path)
            total_frames = metainfo.num_frames // self.hop
            start_frame = torch.randint(total_frames - self.n_frames, [1]).item()
            start_sample = start_frame * self.hop

            wav, _ = torchaudio.load(wav_path, frame_offset=start_sample, num_frames=num_samples)
            speaker = wav_path.split('/')[-2]

        mel = self.to_mel(wav).squeeze(dim=0)
        # print(self.speaker_mapping)
        speaker_id = self.speaker_mapping[speaker]

        return mel, speaker_id

    def __len__(self):
        return len(self.wav_paths)

    def collate_fn(self, batch):
        return torch.utils.data.dataloader.default_collate(batch)


