import json
import os

def hifimanifest_to_vitsmanifest(source_file, target_path):
    samples = [json.loads(sample) for sample in open(source_file)]
    result = []
    missing_files = 0
    for sample in samples:
        tmp = sample["audio_filepath"].split('/')[1::2]
        if "6097" in tmp[0]:
            tmp[0] = "6097_merged"
        path = '/'.join(tmp)
        path = path[:-4] + 'wav'
        path = '/root/storage/TTS/hifi/wavs/' + path
        if not os.path.isfile(path):
            missing_files += 1
            continue
        text = sample["text"]

        result.append('|'.join((path, text)))
    print(missing_files)
    with open(target_path, 'w') as f:
        f.write('\n'.join(result))


def add_speaker(source, target):
    speaker_mapping = {'11614': 0, '11697': 1, '12787': 2, '6097': 3, '6670': 4,
                       '6671': 5, '8051': 6, '9017': 7, '9136': 8, '92': 9}
    samples = [sample.strip() for sample in open(source)]
    samples_with_speaker_id = []
    for sample in samples:
        filepath, text = sample.split("|")
        speaker_id = speaker_mapping[filepath.split('/')[6].split('_')[0]]
        samples_with_speaker_id.append('|'.join((filepath, str(speaker_id), text)))

    with open(target, 'w') as f:
        f.write('\n'.join(samples_with_speaker_id))


if __name__ == "__main__":
    # hifimanifest_to_vitsmanifest("/root/storage/TTS/hifi/merged_manifest_train.txt",
    #                              "/root/storage/TTS/hifi/vits/manifest_train.txt")
    # hifimanifest_to_vitsmanifest("/root/storage/TTS/hifi/merged_manifest_test_dev.txt",
    #                              "/root/storage/TTS/hifi/vits/manifest_test_dev.txt")
    basepath = '/root/storage/TTS/hifi/vits/'
    add_speaker(os.path.join(basepath, "manifest_train_phonemes.txt"),
                os.path.join(basepath, "manifest_train_phonemes_speakers.txt"))
    add_speaker(os.path.join(basepath, "manifest_test_dev_phonemes.txt"),
                os.path.join(basepath, "manifest_test_dev_phonemes_speakers.txt"))
