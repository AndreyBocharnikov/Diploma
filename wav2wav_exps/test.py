# p225, p226, p229 and p232, first 25 samples (because they are alligned)
import json
import os.path
import hydra
import torch

import torchaudio
import tqdm

from train import VQ_VAE

test_speakers = ['p225', 'p226', 'p229', 'p232']


def get_speaker(path):
    return path.split('/')[-2]


def get_sample_number(path):
    return path.split('_')[-2]


def first_30(manifest_4speakers_test, target_path):
    # function takes only from 4speakers manifest
    # ls p225 p226 p229 p232 > 4speakers_test_manifest.txt
    with open(target_path, 'w') as t:
        for sample in open(manifest_4speakers_test):
            sample = sample.strip()
            sample_speaker = get_speaker(sample)
            sample_number = get_sample_number(sample)
            if int(sample_number) < 25 and sample_speaker in test_speakers:
                t.write(sample + '\n')


def convert_test(cfg, test_samples, model_path, speaker_mapping_path, target_path):
    speaker_mapping = json.load(open(speaker_mapping_path))
    checkpoint = torch.load(model_path)
    # print(checkpoint.keys())
    # print(checkpoint['hyper_parameters'])
    model = VQ_VAE(cfg)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    for sample in tqdm.tqdm(open(test_samples).readlines()):
        sample = sample.strip()
        source_speaker = get_speaker(sample)
        sample_number = get_sample_number(sample)
        for target_speaker in test_speakers:
            if source_speaker == target_speaker:
                continue
            wav, sr = torchaudio.load(sample)
            target_speaker_id = torch.LongTensor([speaker_mapping[target_speaker]])
            # print(target_speaker_id.shape)
            converted_wav = model(wav, target_speaker_id)

            converted_wav_path = os.path.join(target_path, source_speaker + '_' + sample_number + '_'
                                              + target_speaker + '.wav')
            torchaudio.save(converted_wav_path, converted_wav.squeeze(dim=0), 16000)


@hydra.main(config_path=".", config_name="vq_vae")
def main(cfg):
    manifest_4speakers_test = '/root/storage/TTS/vctk/manifests/all_samples.txt'
    target_path = '/root/storage/TTS/vctk/manifests/4speakers_first24_test_manifest.txt'
    # first_30(manifest_4speakers_test, target_path)

    model_path = '/root/andrey_b/vq-vae/outputs/parallel/lightning_logs/version_0/checkpoints/last.ckpt'
    generated_path = '/root/storage/TTS/vctk/parallel_generated/'
    speaker_mapping_path = '/root/storage/TTS/vctk/speaker_mapping.txt'
    convert_test(cfg, target_path, model_path, speaker_mapping_path, generated_path)


if __name__ == "__main__":
    main()