import torch
import lightning as L
from liver_tk.nets.unet import SegmentationModel
from liver_tk.datamodule.segmentation_datamodule import SegmentationDataModule
from lightning.pytorch.loggers.mlflow import MLFlowLogger

def train_model():
    torch.set_float32_matmul_precision('medium')
    model = SegmentationModel()
    datamodule = SegmentationDataModule(
        batch_size=1,
        data_root_path="/home/haim/code/tumors/data",
        csv_file_path="/home/haim/code/tumors/liver_tumors/image_and_segment_paths.csv",
        num_workers=1,
    )

    mlflow_logger = MLFlowLogger(
        experiment_name="liver_segmentation",
        tracking_uri="http://127.0.0.1:8080",
        )

    trainer = L.Trainer(
        max_epochs=10, 
        fast_dev_run=True, 
        accelerator="gpu",
        logger=mlflow_logger,
    )
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    train_model()