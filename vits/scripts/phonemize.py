import os
import sys
sys.path.insert(0, '..')
from text.symbols import _letters_ipa, _letters


def get_text_from_manifests(file_path, target_path):
    texts = [line.split('|')[1] for line in open(file_path)]
    with open(target_path, 'w') as f:
        for line in texts:
            f.write(line)


def add_file_path_to_phenemes(manifest_w_words, phonemes_path, target_path):
    with open(manifest_w_words) as f, open(phonemes_path) as g, open(target_path, 'w') as t:
        paths = [sample.split('|')[0] for sample in f]
        phonemes = [phoneme for phoneme in g]
        skip_path_id = 0
        skip_paths = [186504, 246604, 248443, 263187]
        for i in range(len(phonemes)):
            if i + skip_path_id in skip_paths:
                skip_path_id += 1
            t.write('|'.join((paths[i + skip_path_id], phonemes[i])))
        # for sample, phonemes in zip(f, g):
        #     t.write('|'.join((sample.split('|')[0], phonemes)))


def fix_phonemes(sample):
    sample = sample.replace("rɪ^", "rʲ")  # рь и р + гласная это одно и то же
    sample = sample.replace("jj", "j")  # ь + гласная это то же й что и просто й
    sample = sample.replace("u\"", "ʲu").replace("ʲʲ", "ʲ")  # ю после согласной это ьу, а не новый символ
    sample = sample.replace("ʑʲ", "ʑ")  # символ мягкой z итак существует
    sample = sample.replace("tʃʲ", "w")  # ч это не т + мягкая ш
    sample = sample.replace("ʒʲ", "ʒ").replace("ʃʲ", "ʃ").replace("tsʲ", "ts")  # ж, ш, ц всегда твёрдые
    return sample


def make_phoneme_mapping(all_phoneme_sample_path):
    # рь = rɪ^ -> rʲ
    # гласная после ь -> й = jj; й = j; jj -> j
    # ɭu\" -> ɭʲu тюлень говорю вовсю
    # tʃʲ -> w
    all_phonemes = set()
    samples = [sample.strip() for sample in open(all_phoneme_sample_path)]
    print(len(samples))
    phonemized_samples = []
    for j, sample in enumerate(samples):
        # н и л не становятся мягкими когда после них гласные, хотя с и т становятся мягкими когда после них те же ю и
        sample = fix_phonemes(sample)
        # if j == 0:
        #     print(sample)
        phonemes = []
        skip = False
        for i, char in enumerate(sample):
            if skip:
                skip = False
                continue
            if i == len(sample) - 1 or sample[i + 1] != 'ʲ':
                if char == " ":
                    phonemes.append('#')
                    all_phonemes.add('#')
                else:
                    phonemes.append(char)
                    all_phonemes.add(char)
            else:
                phonemes.append(char + '0')
                all_phonemes.add(char + '0')
                skip = True
        # if j == 0:
        #     print(' '.join(phonemes))
        if ' ' in phonemes:
            print(phonemes)
            print(sample)
        phonemized_samples.append(' '.join(phonemes))
    phonemized_all_samples = '\n'.join(phonemized_samples)
    print(len(phonemized_samples))
    # with open("/root/storage/litrec/all_samples_phonemes_fix.txt", 'w') as f:
    #     f.write(phonemized_all_samples)
    # print(all_phonemes)
    print(' ' in all_phonemes)
    print({v: k for k, v in dict([*enumerate(all_phonemes)]).items()})
    # ": ю = u", ^: ???


def check_lu():
    with open("/root/storage/litrec/tacotron_train_short.txt") as f,\
            open("/root/storage/litrec/tacotron_train_short_phonemes.txt") as t:
        cnt_lu_word, cnt_lu_phoneme = 0, 0
        for sample_text, sample_phonems in zip(f, t):
            for word, phonemes in zip(sample_text.split(), sample_phonems.split()):
                if "лю" in word:
                    cnt_lu_word += 1
                if "ɭu\"" in phonemes:
                    cnt_lu_phoneme += 1
                if ("лю" not in word and "ɭu\"" in phonemes) or ("лю" in word and "ɭu\"" not in phonemes):
                    print(word, phonemes)
        print(cnt_lu_word, cnt_lu_phoneme)


def check_english_g2p(phonemes_path):
    phonemes = [phonemes.strip() for phonemes in open(phonemes_path)]
    for i, phoneme in enumerate(phonemes):
        for char in phoneme:
            if char not in _letters_ipa and char not in _letters and char != ' ':
                print(i, char, phoneme)
                break


if __name__ == "__main__":
    # /root/storage/litrec/tacotron_train_short.txt /root/storage/litrec/train_short_texts.txt
    # get_text_from_manifests('/root/storage/TTS/hifi/vits/manifest_train.txt',
    #                         '/root/storage/TTS/hifi/vits/train_texts.txt')
    # get_text_from_manifests('/root/storage/TTS/hifi/vits/manifest_test_dev.txt',
    #                         '/root/storage/TTS/hifi/vits/test_dev_texts.txt')
    # /root/storage/litrec/tacotron_train_short.txt train_short_phonemes_fix.txt tacotron_train_short_phonemes_fix.txt
    basepath = "/root/storage/TTS/hifi/vits"
    add_file_path_to_phenemes(os.path.join(basepath, 'manifest_train.txt'),
                              os.path.join(basepath, "train_phonemes.txt"),
                              os.path.join(basepath, "manifest_train_phonemes.txt"))
    add_file_path_to_phenemes(os.path.join(basepath, 'manifest_test_dev.txt'),
                              os.path.join(basepath, "test_dev_phonemes.txt"),
                              os.path.join(basepath, "manifest_test_dev_phonemes.txt"))

    # make_phoneme_mapping('/root/storage/litrec/all_samples_phonemes.txt')
    # split -l 29584 /root/storage/litrec/all_samples_phonemes_fix.txt
    # если первые train потом val то такой сплит норм, иначе нет :(, чекнуть
    # check_lu()

    # print('train')
    # check_english_g2p("/root/storage/TTS/hifi/vits/train_phonemes.txt")
    # print('dev_test')
    # check_english_g2p("/root/storage/TTS/hifi/vits/test_dev_phonemes.txt")