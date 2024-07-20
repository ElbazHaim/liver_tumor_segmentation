import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import mlflow.pytorch
from monai.networks.nets import UNet
from torchmetrics import MetricCollection, Accuracy
from torchmetrics.detection.iou import IntersectionOverUnion
# from mlflow import log_metric


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        lr=1e-3,
    ):
        super(SegmentationModel, self).__init__()
        self.save_hyperparameters()
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64),
            strides=(2, 2),
            kernel_size=3,
            up_kernel_size=3,
            dropout=0.1,
        )
        metrics = MetricCollection(
            {"IoU": IntersectionOverUnion(num_classes=2), "Accuracy": Accuracy()}
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        preds = torch.argmax(y_hat, dim=1)
        self.train_metrics(preds, y)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        preds = torch.argmax(y_hat, dim=1)
        self.val_metrics(preds, y)
        self.log_dict(self.val_metrics, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # def on_train_epoch_end(self):
    #     metrics = self.train_metrics.compute()
    #     for k, v in metrics.items():
    #         log_metric(k, v)
    #     self.train_metrics.reset()

    # def on_validation_epoch_end(self):
    #     metrics = self.val_metrics.compute()
    #     for k, v in metrics.items():
    #         log_metric(k, v)
    #     self.val_metrics.reset()
