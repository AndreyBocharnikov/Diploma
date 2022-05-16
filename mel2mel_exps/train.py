import itertools

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import LightningDataModule
from losses import FeatureMatchingLoss, DiscriminatorLoss, GeneratorLoss
from modules.mel_to_wav_modules import MultiPeriodDiscriminator,MultiScaleDiscriminator
from utils import load_vocoder, log_wavs, ComputeMel


class VQVAE(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vq_vae = instantiate(cfg.model.vq_vae)
        self.generator = instantiate(cfg.model.generator)

        self.msd = MultiScaleDiscriminator()
        self.mpd = MultiPeriodDiscriminator()

        self.to_mel = ComputeMel(**cfg.model.mel_cfg)

        self.reconstruction_loss = torch.nn.L1Loss()
        self.feature_loss = FeatureMatchingLoss()
        self.discriminator_loss = DiscriminatorLoss()
        self.generator_loss = GeneratorLoss()

        self.l1_factor = cfg.model.l1_factor
        self.latent_loss_weight = cfg.model.latent_loss_weight
        self.sample_rate = cfg.sample_rate

        self.automatic_optimization = False

    def configure_optimizers(self):
        self.optim_g = instantiate(self.cfg.optim, params=itertools.chain(self.vq_vae.parameters(), self.generator.parameters()))
        self.optim_d = instantiate(self.cfg.optim, params=itertools.chain(self.msd.parameters(), self.mpd.parameters()))
        return self.optim_g, self.optim_d

    def training_step(self, batch, batch_idx):
        source_wav, source_mel, target_wav, target_mel = batch

        vae_embed, latent_loss = self.vq_vae(source_mel)
        pred_wav = self.generator(vae_embed)

        pred_mel = self.to_mel(pred_wav.squeeze(dim=1))

        self.optim_d.zero_grad()
        mpd_score_real, mpd_score_gen, _, _ = self.mpd(target_wav, pred_wav.detach())
        loss_disc_mpd = self.discriminator_loss(mpd_score_real, mpd_score_gen)

        msd_score_real, msd_score_gen, _, _ = self.msd(target_wav, pred_wav.detach())
        loss_disc_msd = self.discriminator_loss(msd_score_real, msd_score_gen)
        loss_d = loss_disc_msd + loss_disc_mpd
        self.manual_backward(loss_d)
        self.optim_d.step()

        self.optim_g.zero_grad()
        loss_mel = self.reconstruction_loss(pred_mel, target_mel)
        _, mpd_score_gen, fmap_mpd_real, fmap_mpd_gen = self.mpd(target_wav, pred_wav)
        _, msd_score_gen, fmap_msd_real, fmap_msd_gen = self.msd(target_wav, pred_wav)
        loss_fm_mpd = self.feature_loss(fmap_r=fmap_mpd_real, fmap_g=fmap_mpd_gen)
        loss_fm_msd = self.feature_loss(fmap_r=fmap_msd_real, fmap_g=fmap_msd_gen)
        loss_gen_mpd = self.generator_loss(disc_outputs=mpd_score_gen)
        loss_gen_msd = self.generator_loss(disc_outputs=msd_score_gen)
        loss_g = loss_gen_msd + loss_gen_mpd + loss_fm_msd + loss_fm_mpd + \
                 loss_mel * self.l1_factor + self.latent_loss_weight * latent_loss.mean()
        self.manual_backward(loss_g)
        self.optim_g.step()

        metrics = {
            "reconstruction_loss": loss_mel.item(),
            "latent_loss": latent_loss.item(),
            "adv gen loss": loss_gen_mpd.item() + loss_gen_msd.item(),
            "adv disc loss": loss_d.item(),
            "fml": loss_fm_msd.item() + loss_fm_mpd.item()
        }
        self.log_dict(metrics, on_step=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            source_wavs, source_mels, mel_lens = batch
            converted_mels, _ = self.vq_vae(source_mels)
            converted_wavs = self.generator(converted_mels)

            handle = "converted wav"
            for i, converted_wav in enumerate(converted_wavs):
                self.logger.experiment.add_audio(handle + f" {str(i)}", converted_wav.detach().cpu().numpy(),
                                                 global_step=self.global_step,
                                                 sample_rate=self.sample_rate)

            if self.global_step == 0:
                handle = "gt wav"
                for i, source_wav in enumerate(source_wavs):
                    self.logger.experiment.add_audio(handle + f" {str(i)}", source_wav.detach().cpu().numpy(),
                                                     global_step=self.global_step,
                                                     sample_rate=self.sample_rate)


@hydra.main(config_path="configs", config_name="mel_to_wav") # vq_vae3
def main(cfg):
    checkpoint_callback = ModelCheckpoint(
        monitor="reconstruction_loss",
        filename="{epoch:03d}-{reconstruction_loss:.3f}",
        save_top_k=2,
        mode="min",
        save_last=True
    )
    trainer = pl.Trainer(callbacks=[checkpoint_callback], **cfg.trainer)
    model = VQVAE(cfg=cfg)
    data_module = LightningDataModule(cfg)
    """
    if cfg.model.pretrained_path is not None:
        print('loading from pretrain')
        checkpoint = torch.load(cfg.model.pretrained_path)
        model.load_state_dict(checkpoint["state_dict"])
    """
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()