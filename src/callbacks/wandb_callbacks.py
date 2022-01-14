from pytorch_lightning import Trainer, Callback
from pytorch_lightning .loggers import WandbLogger, LoggerCollection
import wandb


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )

def get_batch_images_to_log(trainer, pl_module, batch):
    val_imgs, val_labels = batch
    val_imgs = [x.to(device=pl_module.device) for x in val_imgs]

    # val_imgs = val_imgs.to(device=pl_module.device)
    
    box_preds = pl_module(val_imgs)

    images_to_log = []
    for index, (boxes, img) in enumerate(zip(box_preds, val_imgs)):
        bbs = boxes["boxes"]
        labels = boxes["labels"]
        scores = boxes["scores"]
        positions = []
        # for box, label in zip(bbs, labels):
        for i in range(len(labels)):
            box = bbs[i]
            label = labels[i]
            score = scores[i]
            box_pos = {
                "position": {
                    "minX": int(box[0]),
                    "minY": int(box[1]),
                    "maxX": int(box[2]),
                    "maxY": int(box[3])
                },
                "class_id": int(label),
                "scores": {"confidence": float(score)},
                "box_caption": str(int(label)) + ":" + str(round(float(score), 3)),
                "domain": "pixel"
            }
            positions.append(box_pos)
        bbs_gt = val_labels[index]["boxes"]
        labels_gt = val_labels[index]["labels"]
        gt_positions = []
        for box, label in zip(bbs_gt, labels_gt):
            box_pos = {
                "position": {
                    "minX": int(box[0]),
                    "minY": int(box[1]),
                    "maxX": int(box[2]),
                    "maxY": int(box[3])
                },
                "class_id": int(label),
                "domain": "pixel"
            }
            gt_positions.append(box_pos)

        boxes_data = {
            "predictions": {
                "box_data": positions
            },
            "ground_truth": {
                "box_data": gt_positions
            }

        }
        wandb_img = wandb.Image(img, boxes=boxes_data)
        images_to_log.append(wandb_img)
    return images_to_log


class LogImagePredictionsDetection(Callback):
    def __init__(self):
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        self.ready = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:

            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment
            # val_samples = next(iter(trainer.datamodule.val_dataloader()))
            val_samples = next(iter(trainer.val_dataloaders[0]))
            images_to_log = get_batch_images_to_log(trainer, pl_module, val_samples)
            experiment.log({"val/image_sample": images_to_log})

class LogImagePredictionsDetectionFull(Callback):
    def __init__(self, interval = 30):
        self.ready = True
        self.interval = interval
        self.index = 0

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        self.ready = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            self.index += 1
            if self.index % self.interval == 0:

                logger = get_wandb_logger(trainer=trainer)
                experiment = logger.experiment
                # val_samples = next(iter(trainer.datamodule.val_dataloader()))
                for index, batch in enumerate(trainer.val_dataloaders[0]):
                    if index == 8:
                        break
                    images_to_log = get_batch_images_to_log(trainer, pl_module, batch)
                    experiment.log({"val/full_images"+str(index): images_to_log})

