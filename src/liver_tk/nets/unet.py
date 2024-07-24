import torch
import torch.nn.functional as F
import lightning as L
from monai.networks.nets import UNet
# from monai.metrics import DiceMetric,
# from torchmetrics import MetricCollection
# from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore

from icecream import ic


class SegmentationModel(L.LightningModule):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: tuple = (16, 32, 64),
        strides: tuple = (2, 2),
        lr: float = 1e-3,
        num_classes: int = 1,
    ):
        super(SegmentationModel, self).__init__()
        self.save_hyperparameters()
        self.model = UNet(
            spatial_dims=3,
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.out_channels,
            channels=self.hparams.channels,
            strides=self.hparams.strides,
            kernel_size=3,
            up_kernel_size=3,
            dropout=0.1,
        )
        # metrics = MetricCollection(
        #     {
        #         "IoU": MeanIoU(num_classes=self.hparams.num_classes),
        #         "GeneralizedDiceScore": GeneralizedDiceScore(
        #             num_classes=self.hparams.num_classes,
        #         ),
        #     }
        # )
        # self.train_metrics = metrics.clone(prefix="train_")
        # self.val_metrics = metrics.clone(prefix="val_")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.to(dtype=torch.float16)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        ic(y_hat.dtype)
        ic(y_hat.shape)
        ic(y.dtype)
        ic(y.shape)
        # self.train_metrics(y_hat, y)
        # self.log_dict(self.train_metrics, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.to(dtype=torch.float16)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        # self.val_metrics(y_hat, y)
        # self.log_dict(self.val_metrics, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
