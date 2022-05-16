import random
from pathlib import Path
import torchaudio
import numpy as np
import os
from scipy.io.wavfile import read
import tqdm
import torch
import youtokentome as yttm
import re
import numpy as np


global_paths = ['1k_short_keywords/clear_transcripts.txt', '1k_short_no_keywords/clear_transcripts.txt',
         '32k_long_no_keywords/transcripts.txt', '9k_short_no_keywords/transcripts.txt',
         '4k_long_keywords/clear_transcripts.txt', '4k_long_no_keywords/clear_transcripts.txt']
global_paths = ['4k_long_no_keywords/clear_transcripts.txt']
basepath = '/root/storage/vk_comments/'
paths = list(map(lambda x: os.path.join(basepath, x), global_paths))


def resample(basepath):
    oggs = list(Path(basepath).glob('**/*.ogg'))

    for flac_path in tqdm.tqdm(oggs):
        name = str(flac_path).split('/')[-1].split('.')[0] + '.wav'
        current_wav_path = os.path.join(wav_path, name)
        os.system(f"sox {flac_path} {current_wav_path}")


def check_ids():
    sample_paths = list([sample.split()[0] for path in paths for sample in open(path)])
    print(len(sample_paths))
    print(len(set(sample_paths)))


def check_names():
    names = set([name.strip() for name in open('/root/storage/vk_comments/names.txt')])
    cnt = 0
    first_words = list([sample.split()[1] for path in paths for sample in open(path) if len(sample.split()) > 2])
    print(len(names), len(first_words))
    for first_word in first_words:
        if first_word in names:
            cnt += 1
    print(cnt / len(first_words))


scripted_model = torch.jit.load("/root/storage/vk_comments/models/scripted_punk.pth")
device = torch.device("cuda")
bpe = yttm.BPE("/root/storage/vk_comments/models/chat.yttm")
i2c = ["0ß", "0-", "0.", "0:", "0!", "0,", "0?",
   "1ß", "1-", "1.", "1:", "1!", "1,", "1?"]
c2i = {c: i for i, c in enumerate(i2c)}


def apply_capitalization(bpe_subword):
    if bpe_subword[0]=='▁':
        return '▁'+bpe_subword[1:-1].capitalize() if bpe_subword[-1] == '1' else bpe_subword[:-1]
    return bpe_subword[:-1].lower()


def replace_unnesessary_punk(text):
    return re.sub('[^\w\s]', '', text[:-1])+text[-1]


def add_punk_new_jit(text):
    input = torch.tensor(bpe.encode(text)).view(1, -1).to(device)
    generations = scripted_model(input).argmax(-1)
    text=''.join([apply_capitalization(token + punk[::-1]) for token, punk in zip(
    [bpe.id_to_subword(int(idx)) for idx in input.cpu().numpy()[0]],
    [i2c[int(idx-1)] for idx in generations.cpu().numpy()[0]])]).replace('ß','').split('▁')[1:]
    return ' '.join([replace_unnesessary_punk(tmp) for tmp in text]).replace('- ','-')


def add_punctuation():
    names = set([name.strip() for name in open('/root/storage/vk_comments/names.txt')])
    target_path = '/root/storage/vk_comments/punk.txt'
    paths = ['/root/storage/vk_comments/4k_long_no_keywords/clear_transcripts.txt',
             '/root/storage/vk_comments/32k_long_no_keywords/transcripts.txt']
    skipped_by_name, skipped_by_empty_text = 0, 0
    with open(target_path, 'w') as t:
        for path in paths:
            samples = list(open(path).readlines())
            print(len(samples))
            for i, sample in enumerate(tqdm.tqdm(samples)):
                tmp = sample.strip().split()
                name = tmp[0]
                skip_by_name = len(tmp) > 1 and tmp[1] in names and random.random() < 0.9
                if len(tmp) < 2 or skip_by_name:
                    skipped_by_name += skip_by_name
                    skipped_by_empty_text += len(tmp) < 2
                    continue
                text = ' '.join(tmp[1:])
                # print(text)
                text_w_punk = add_punk_new_jit(text)
                t.write(name + ' ' + text + '\n' + name + ' ' + text_w_punk + '\n\n')
    print(skipped_by_name, skipped_by_empty_text)


def check_sum_durations_short_and_long():
    short_path = '/root/storage/vk_comments/short_ids.txt'
    long_path = '/root/storage/vk_comments/32&4_long_no_keywords_w_punk.txt'
    p = list(Path(basepath).glob('**/*.wav'))
    durations = dict()
    print('computing all durations')
    for path in tqdm.tqdm(p):
        name = str(path).split('/')[-1].split('.')[0]
        metadata = torchaudio.info(str(path))
        duration = metadata.num_frames / metadata.sample_rate
        durations[name] = duration

    # COMPUTE SHORT DUR
    short_samples = [sample.strip() for sample in open(short_path)]
    short_sum_duration = sum([durations[short_sample] for short_sample in short_samples if short_sample in durations])
    print('short sum', short_sum_duration)

    # COMPUTE LONG DUR
    long_samples = []
    for sample in open(long_path):
        sample = sample.strip()
        if sample == '----------':
            break
        long_samples.append(sample.split()[0])
    long_sum_duration = sum([durations[long_sample] for long_sample in long_samples if long_sample in durations])
    print('long sum', long_sum_duration)


def merge_short_and_long():
    short_ids_path = '/root/storage/vk_comments/short_ids.txt'
    short_sample_path = '/root/storage/vk_comments/9k_short_no_keywords/transcripts.txt'
    long_sample_path = '/root/storage/vk_comments/32&4_long_no_keywords_w_punk.txt'
    merged_samples = []
    merged_samples_path = '/root/storage/vk_comments/merged_samples.txt'
    short_id_sample_mapping = dict()
    for sample in open(short_sample_path):
        tmp = sample.strip().split()
        name = tmp[0]
        text = ' '.join(tmp[1:])
        short_id_sample_mapping[name] = text

    for sample in open(short_ids_path):
        name = sample.strip()
        if name == '':
            continue
        merged_samples.append(' '.join((name, short_id_sample_mapping[name])))

    for sample in open(long_sample_path):
        sample = sample.strip()
        if sample == "----------":
            break
        merged_samples.append(sample)

    with open(merged_samples_path, 'w') as f:
        f.write('\n'.join(merged_samples))


def check_lens():
    people_names = set([name.strip() for name in open('/root/storage/vk_comments/names.txt')])

    name_text_mapping = dict()
    for path in paths:
        for sample in open(path):
            tmp = sample.strip().split()
            name = tmp[0]
            text = ' '.join(tmp[1:])
            name_text_mapping[name] = text

    p = list(Path(basepath).glob('**/*.wav'))
    durations, texts = [], []
    for path in tqdm.tqdm(p):
        name = str(path).split('/')[-1].split('.')[0]
        if name not in name_text_mapping:
            continue
        text = name_text_mapping[name]
        if len(text.split()) == 0:
            continue
        people_name = text.split()[0]
        if people_name in people_names and random.random() < 0.95:
            continue
        wav, sr = torchaudio.load(str(path))
        # full_path = '/'.join(str(path).split('/')[:-1])
        # wav_name = str(path).split('/')[-1].split('.')[0] + '.wav'
        # print(os.path.join(full_path, wav_name))
        # torchaudio.save(os.path.join(full_path, wav_name), wav, sr)
        dur = wav.shape[1] / sr
        if dur > 11.5 or len(text) > 190:
            continue
        durations.append(dur)
        texts.append(len(text))
    print(sum(durations))
    print(len(texts))
    print(np.mean(durations), np.median(durations))
    print(np.mean(texts), np.median(texts))


if __name__ == "__main__":
    # check_ids()
    # check_lens()
    # check_names()
    # add_punctuation()
    # check_sum_durations_short_and_long()
    merge_short_and_long()


# TODO по id достают только их (9k_short_no_keywords)
# для 4k_long_no_keywords оставить только часть с именами