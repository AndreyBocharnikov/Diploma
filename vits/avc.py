import json
import os.path
from collections import defaultdict

import tqdm

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader

import utils
from data_utils import TextAudioLoader, TextAudioCollate, AVCSynthesizer
from models import SynthesizerTrn
from text import symbols


def rename_ljs_filelist(real_path, source_files, target_filelists):
    for source_file, target_filelist in zip(source_files, target_filelists):
        with open(source_file) as f, open(target_filelist, 'w') as t:
            for sample in f:
                path, phonemes = sample.strip().split('|')
                sample_name = path.split('/')[1]
                t.write('|'.join((real_path + sample_name, phonemes)) + '\n')


def load_checkpoint(model, checkpoint_path):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            raise ValueError("%s is not in the checkpoint" % k)
            # new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    print("Loaded checkpoint '{}' )".format(checkpoint_path))
    return model


def get_ljspeech_durations(h_params, save_path, pretrained_ljs_path, generated_durs=False, generated_wavs_path=None):
    net_g = SynthesizerTrn(
        len(symbols),
        h_params.data.filter_length // 2 + 1,
        h_params.train.segment_size // h_params.data.hop_length,
        **h_params.model).cuda()

    load_checkpoint(net_g, pretrained_ljs_path)
    # net_g.eval()

    train_dataset = TextAudioLoader(h_params.data.training_files, h_params.data)
    collate_fn = TextAudioCollate(return_paths=h_params.data.load_path)
    train_loader = DataLoader(train_dataset, num_workers=8, shuffle=True, pin_memory=True,
                              collate_fn=collate_fn, persistent_workers=True, batch_size=32)
    for batch in tqdm.tqdm(train_loader):
        x, x_lengths, spec, spec_lengths, y, y_lengths, paths = batch
        x, x_lengths = x.cuda(non_blocking=True), x_lengths.cuda(non_blocking=True)
        spec, spec_lengths = spec.cuda(non_blocking=True), spec_lengths.cuda(non_blocking=True)
        y, y_lengths = y.cuda(non_blocking=True), y_lengths.cuda(non_blocking=True)

        with torch.no_grad():
            if generated_durs:
                generated_wavs, _, _, _, phoneme_lengths = net_g.infer(x, x_lengths, return_durs=True)
            else:
                phoneme_lengths = net_g.get_model_durations(x, x_lengths, spec, spec_lengths)
            generated_mel_lens = torch.clamp_min(torch.sum(phoneme_lengths, [1, 2]), 1).long()
            phoneme_lengths = phoneme_lengths.squeeze(dim=1).cpu().numpy()
            for i, (sample_path, phoneme_length) in enumerate(zip(paths, phoneme_lengths)):
                sample_name = sample_path.split('/')[-1].split('.')[0]
                tgt_path = os.path.join(save_path, sample_name)
                np.save(tgt_path, phoneme_length[:x_lengths[i]])

                # print(sample_path)
                # print(sample_name)
                if generated_durs:
                    generated_sample_path = os.path.join(generated_wavs_path, sample_name + '.wav')
                    torchaudio.save(generated_sample_path, generated_wavs[i, :, :generated_mel_lens[i] * 256].cpu(), 22050,
                                    encoding="PCM_S", bits_per_sample=16)


def get_speaker_to_speakerid_mapping(speaker_ids_path):
    list_op_speakers = [sample.strip() for sample in open('filelists/avc_vctk_speakers.txt')]
    print(len(list_op_speakers))
    print(list_op_speakers)
    vctk_train_filelist = 'filelists/vctk_audio_sid_text_train_filelist.txt.cleaned'
    speaker_mapping = dict()
    speaker_cnt = defaultdict(int)
    with open(vctk_train_filelist) as f:
        for sample in f:
            path, speaker_id, _ = sample.strip().split('|')
            speaker = path.split('/')[1]
            if speaker not in list_op_speakers:
                continue
            speaker_cnt[speaker] += 1
            speaker_mapping[speaker] = int(speaker_id)
    print(len(speaker_mapping))
    for speaker in list_op_speakers:
        if speaker not in speaker_mapping:
            print("!!!", speaker)
    print(speaker_cnt)
    print(speaker_mapping)

    with open(speaker_ids_path, 'w') as t:
        for speaker_id in speaker_mapping.values():
            t.write(str(speaker_id) + '\n')

    speaker_mapping_path = '/root/andrey_b/vits/filelists/avc_vctk_speaker_mapping.txt'
    with open(speaker_mapping_path, 'w') as f:
        f.write(json.dumps(speaker_mapping))


def generate_align_dataset(h_params, speakers_to_generate_ids, pretrained_vctk_path, generated_wavs_path):
    speakers_list = [int(speaker.strip()) for speaker in open(speakers_to_generate_ids)]
    for speaker_id in speakers_list:
        os.makedirs(os.path.join(generated_wavs_path, str(speaker_id)), exist_ok=True)
    speakers = torch.LongTensor(speakers_list).cuda()

    bsz = speakers.shape[0]

    net_g = SynthesizerTrn(
        len(symbols),
        h_params.data.filter_length // 2 + 1,
        h_params.train.segment_size // h_params.data.hop_length,
        n_speakers=h_params.data.n_speakers,
        **h_params.model).cuda()

    load_checkpoint(net_g, pretrained_vctk_path)
    net_g.eval()

    audiopaths_text = '/root/andrey_b/vits/filelists/ljs_correct_train.txt'
    duration_path = '/root/storage/TTS/avc/ljs_phoneme_durs_generated'
    dataset = AVCSynthesizer(audiopaths_text, duration_path)
    for i, sample in enumerate(tqdm.tqdm(dataset)):
        duration, text, audiopath = sample
        text_lengths = torch.LongTensor(bsz * [text.shape[0]]).cuda()
        text = text.unsqueeze(dim=0).repeat((bsz, 1)).cuda()
        duration = duration.unsqueeze(dim=0).repeat((bsz, 1)).cuda()

        with torch.no_grad():
            wavs, spec_mask = net_g.generate_with_duration(text, text_lengths, speakers, duration)
            for wav, speaker_id in zip(wavs, speakers):
                sample_name = audiopath.split('/')[-1].split('.')[0]
                generated_sample_path = os.path.join(generated_wavs_path, str(speaker_id.cpu().item()),
                                                     sample_name + '.wav')
                # print(generated_sample_path)
                torchaudio.save(generated_sample_path, wav.cpu(), 22050, encoding="PCM_S", bits_per_sample=16)


if __name__ == "__main__":
    # target_filelists = ['/root/andrey_b/vits/filelists/ljs_correct_train.txt',
    #                     '/root/andrey_b/vits/filelists/ljs_correct_test.txt']
    # source_files = ['/root/andrey_b/vits/filelists/ljs_audio_text_train_filelist.txt.cleaned',
    #                 '/root/andrey_b/vits/filelists/ljs_audio_text_test_filelist.txt.cleaned']
    # rename_ljs_filelist('/root/storage/TTS/LJSpeech-1.1/wavs/', source_files, target_filelists)

    # h_params = utils.get_hparams()
    save_path = '/root/storage/TTS/avc/ljs_phoneme_durs_generated/'
    generated_ljs_wavs_path = '/root/storage/TTS/avc/generated_ljs/'
    pretrained_ljs_path = '/root/storage/TTS/avc/pretrained_models/pretrained_ljs.pth'
    # get_ljspeech_durations(h_params, save_path, pretrained_ljs_path, True, generated_ljs_wavs_path)

    # avc_vctk_info.txt ручками
    # awk '{print $2}' avc_vctk_info.txt > avc_vctk_speakers.txt

    speaker_ids_path = 'filelists/avc_vctk_speaker_ids.txt'
    get_speaker_to_speakerid_mapping(speaker_ids_path)

    speakers_to_generate_ids = 'filelists/avc_vctk_speaker_ids.txt'
    pretrained_vctk_path = '/root/storage/TTS/avc/pretrained_models/pretrained_vctk.pth'
    generated_wavs_path = '/root/storage/TTS/avc/generated_wavs_w_generated_durs'
    # generate_align_dataset(h_params, speaker_ids_path, pretrained_vctk_path, generated_wavs_path)

"""
1 p225  23  F    English    Southern  England
2 p226  22  M    English    Surrey
3 p227  38  M    English    Cumbria
4 p257  24  F    English    Southern  England
5 p233  23  F    English    Staffordshire
6 p236  23  F    English    Manchester
7 p243  22  M    English    London
8 p244  22  F    English    Manchester
9 p267  23  F    English    Yorkshire
10 p268  23  F    English    Southern  England
11 p269  20  F    English    Newcastle
12 p270  21  M    English    Yorkshire
13 p273  23  M    English    Suffolk
14 p274  22  M    English    Essex
15 p276  24  F    English    Oxford
16 p277  23  F    English    NE  England
17 p278  22  M    English    Cheshire
18 p279  23  M    English    Leicester
19 p286  23  M    English    Newcastle
20 p287  23  M    English    York
21 p294  33  F    American  San  Francisco
22 p299  25  F    American  California
23 p301  23  F    American  North  Carolina
24 p362  29  F    American 
25 p334  18  M    American  Chicago
26 p360  19  M    American  New  Jersey
27 p311  21  M    American  Iowa
28 p345  22  M    American  Florida
29 p248  23  F    Indian  
30 p376  22  M    Indian

p225, p233, p236  p226 p227 p243 p267 p270
p228  22  F    English    Southern  England
p232  23  M    English    Southern  England
p239  22  F    English    SW  England
p259  23  M    English    Nottingham
"""