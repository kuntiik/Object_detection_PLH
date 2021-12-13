import hydra
from omegaconf import DictConfig
from pl_bolts import datamodules
from pytorch_lightning import (Callback, LightningDataModule, LightningModule, Trainer, seed_everything)

from pytorch_lightning.loggers import LightningLoggerBase
from pl_bolts.datamodules import MNISTDataModule

from src.utils import utils

log = utils.get_logger(__name__)

def train(config : DictConfig):
    if config.get("seed"):
        seed_everything(config.seed, workers=True)
    
    log.info(f"Instantiating module <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model)

    callbacks = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))


    logger = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))
    
    trainer = hydra.utils.instantiate(config.trainer, logger=logger, callbacks=callbacks)

    # dm = MNISTDataModule(num_workers=4, pin_memory=False)
    dm = hydra.utils.instantiate(config.datamodule)

    log.info("Starting training")
    trainer.fit(model=model, datamodule=dm)