import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.utils import instantiate


class SpeakerClassification(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = instantiate(cfg.model)
        self.loss = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        self.optim = instantiate(self.cfg.optim, params=self.model.parameters())
        return self.optim

    def forward(self, batch):
        mels, labels = batch
        predictions = self.model(mels)

        loss = self.loss(predictions, labels)
        accuracy = (torch.argmax(predictions, dim=1) == labels).float().mean()
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self(batch)

        metrics = {
            "train loss": loss.item(),
            "train accuracy": accuracy
        }
        self.log_dict(metrics, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self(batch)

        metrics = {
            "val loss": loss.item(),
            "val_accuracy": accuracy
        }
        self.log_dict(metrics, on_step=True, sync_dist=True)
