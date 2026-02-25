import os
import lightning as L
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
from lightning.pytorch.strategies.deepspeed import DeepSpeedStrategy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, TQDMProgressBar
from transformers import DataCollatorForSeq2Seq

from module import LLamaFTLightningModule, FTDataModule


os.environ["TOKENIZERS_PARALLELISM"] = "0"


if __name__ == "__main__":
    L.pytorch.cli_lightning_logo()
    training_args = LightningArgumentParser()
    cli = LightningCLI(
        model_class=LLamaFTLightningModule,
        datamodule_class=FTDataModule,
        seed_everything_default=42,
        trainer_defaults={
            "reload_dataloaders_every_n_epochs": 1,
            "strategy": "deepspeed",
            "precision": "bf16-mixed",
            "accumulate_grad_batches": 4,
            "profiler": "PassThroughProfiler",
            "logger": [WandbLogger(project="LLM-Finetuning"),],
            "callbacks": [EarlyStopping(monitor="val_loss", patience=3), 
                          LearningRateMonitor(),
                          TQDMProgressBar(refresh_rate=30)],
        },
        save_config_callback=None)
    # cli.add_arguments_to_parser(training_args)
