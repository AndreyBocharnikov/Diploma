import hydra
import torchaudio
from hydra.utils import instantiate

from utils import load_vocoder, ComputeMel


@hydra.main(config_path="configs", config_name="vq_vae")
def main(cfg):
    # check_wav_path = '/root/storage/TTS/avc/generated_wavs/0/LJ017-0069.wav'
    check_wav_path = '/root/storage/TTS/LJSpeech-1.1/wavs/LJ017-0069.wav'
    generated_wav_path = '/root/andrey_b/vc/avc/generated/0069_g4.wav'
    wav, sr = torchaudio.load(check_wav_path)
    print(wav.shape, )

    to_mel = instantiate(cfg.model.to_mel)
    mel = to_mel(wav)
    print(mel.shape)

    vocoder = load_vocoder(cfg.model)
    generated_wavs = vocoder(mel)
    print(generated_wavs.shape)
    torchaudio.save(generated_wav_path, generated_wavs.squeeze(dim=0), 22050)


if __name__ == "__main__":
    main()