import itertools

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from nemo.collections.tts.losses.hifigan_losses import FeatureMatchingLoss, DiscriminatorLoss, GeneratorLoss
from nemo.collections.tts.modules.hifigan_modules import MultiScaleDiscriminator, MultiPeriodDiscriminator
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.functional as F

from dataset import LightningDataModule
from model import VQ_VAEModel
from utils import ComputeMel


class VQ_VAE(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vq_vae = VQ_VAEModel(cfg.model)

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
        self.optim_g = instantiate(self.cfg.optim, params=self.vq_vae.parameters())
        self.optim_d = instantiate(self.cfg.optim, params=itertools.chain(self.msd.parameters(), self.mpd.parameters()))
        return self.optim_g, self.optim_d

    def forward(self, wav, speaker):
        converted_wavs, _ = self.vq_vae(wav.unsqueeze(dim=0), speaker)
        return converted_wavs

    def training_step(self, batch, batch_idx):
        source_wavs, target_wavs, speakers = batch
        converted_wavs, latent_loss = self.vq_vae(source_wavs.unsqueeze(dim=1), speakers)

        source_mel = self.to_mel(source_wavs)
        source_wavs = source_wavs.unsqueeze(1)
        converted_wavs_mel = self.to_mel(converted_wavs.squeeze(1))

        # train discriminator
        self.optim_d.zero_grad()
        mpd_score_real, mpd_score_gen, _, _ = self.mpd(y=source_wavs, y_hat=converted_wavs.detach())
        loss_disc_mpd, _, _ = self.discriminator_loss(
            disc_real_outputs=mpd_score_real, disc_generated_outputs=mpd_score_gen
        )
        msd_score_real, msd_score_gen, _, _ = self.msd(y=source_wavs, y_hat=converted_wavs.detach())
        loss_disc_msd, _, _ = self.discriminator_loss(
            disc_real_outputs=msd_score_real, disc_generated_outputs=msd_score_gen
        )
        loss_d = loss_disc_msd + loss_disc_mpd
        self.manual_backward(loss_d)
        self.optim_d.step()

        # train generator
        self.optim_g.zero_grad()
        loss_mel = F.l1_loss(converted_wavs_mel, source_mel)
        _, mpd_score_gen, fmap_mpd_real, fmap_mpd_gen = self.mpd(y=source_wavs, y_hat=converted_wavs)
        _, msd_score_gen, fmap_msd_real, fmap_msd_gen = self.msd(y=source_wavs, y_hat=converted_wavs)
        loss_fm_mpd = self.feature_loss(fmap_r=fmap_mpd_real, fmap_g=fmap_mpd_gen)
        loss_fm_msd = self.feature_loss(fmap_r=fmap_msd_real, fmap_g=fmap_msd_gen)
        loss_gen_mpd, _ = self.generator_loss(disc_outputs=mpd_score_gen)
        loss_gen_msd, _ = self.generator_loss(disc_outputs=msd_score_gen)
        latent_loss = latent_loss.mean()
        loss_g = loss_gen_msd + loss_gen_mpd + loss_fm_msd + loss_fm_mpd + \
                 loss_mel * self.l1_factor + latent_loss * self.latent_loss_weight
        self.manual_backward(loss_g)
        self.optim_g.step()

        metrics = {
            "fml": loss_fm_mpd.item() + loss_fm_msd.item(),
            "adv_gen": loss_gen_mpd.item() + loss_gen_msd.item(),
            "g_loss": loss_mel.item(),
            "adv_disc": loss_disc_mpd.item() + loss_disc_msd.item(),
            "latent loss": latent_loss.item()
        }

        self.log_dict(metrics, on_step=True, sync_dist=True)
        self.log("g_l1_loss", loss_mel, prog_bar=True, logger=False, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        source_wavs, target_wavs, speakers = batch
        converted_wavs, _ = self.vq_vae(source_wavs.unsqueeze(dim=1), speakers)
        wavs_mel = self.to_mel(source_wavs)
        converted_wavs_mel = self.to_mel(converted_wavs.squeeze(1))
        loss_mel = F.l1_loss(converted_wavs_mel, wavs_mel)

        self.log_dict({"reconstruction_loss": loss_mel}, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            # converted_wavs, _ = self.vq_vae(wavs.unsqueeze(dim=1), (speakers + 1) % self.cfg.model.n_speakers)

            handle = "converted wav"
            for i, converted_wav in enumerate(converted_wavs):
                self.logger.experiment.add_audio(handle + f" {str(i)}", converted_wav.detach().cpu().numpy(),
                                                 global_step=self.global_step,
                                                 sample_rate=self.sample_rate)

            handle = "target wav"
            for i, target_wav in enumerate(target_wavs):
                self.logger.experiment.add_audio(handle + f" {str(i)}", target_wav.detach().cpu().numpy(),
                                                 global_step=self.global_step,
                                                 sample_rate=self.sample_rate)
            """
            if self.global_step == 0:
                handle = "source wav"
                for i, source_wav in enumerate(wavs):
                    self.logger.experiment.add_audio(handle + f" {str(i)}", source_wav.detach().cpu().numpy(),
                                                     global_step=self.global_step,
                                                     sample_rate=self.sample_rate)
            """


@hydra.main(config_path=".", config_name="vq_vae")
def main(cfg):
    checkpoint_callback = ModelCheckpoint(
        monitor="reconstruction_loss",
        filename="{epoch:03d}-{reconstruction_loss:.3f}",
        save_top_k=2,
        mode="min",
        save_last=True
    )
    trainer = pl.Trainer(callbacks=[checkpoint_callback], **cfg.trainer)
    model = VQ_VAE(cfg=cfg)
    data_module = LightningDataModule(cfg)
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()
