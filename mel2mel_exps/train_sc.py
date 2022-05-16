from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import hydra

from dataset import LightningDataModule
from models.speaker_classification import SpeakerClassification


@hydra.main(config_path="configs", config_name="speaker_rec") # vq_vae3
def main(cfg):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy",
        filename="{epoch:03d}-{val_accuracy:.3f}",
        save_top_k=2,
        mode="max",
        save_last=True
    )
    trainer = pl.Trainer(callbacks=[checkpoint_callback], **cfg.trainer)
    model = SpeakerClassification(cfg=cfg)
    data_module = LightningDataModule(cfg)
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()