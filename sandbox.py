import pandas as pd
from pl_bolts.models.detection.faster_rcnn import FasterRCNN
from pytorch_lightning import Trainer, callbacks
import timm
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
from typing import List, Any
import matplotlib.pyplot as plt
import numpy as np
import  matplotlib.patches as patches
import torchvision
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger, LoggerCollection
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2

# timm.list_models('*efficientnetv2*')
train_root : str = "/datagrid/public_datasets/COCO/train2017"
val_root : str = "/datagrid/public_datasets/COCO/val2017"
train_annotations : str = "/datagrid/public_datasets/COCO/annotations/instances_train2017.json"
val_annotations : str = "/datagrid/public_datasets/COCO/annotations/instances_val2017.json"

def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x+w, y+h]

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

class LogImagePredictions(Callback):

    # def on_train_epoch_end(self, trainer, pl_module):
    #     logger = get_wandb_logger(trainer=trainer)
    #     experiment = logger.experiment

    #     # val_samples = next(iter(trainer.datamodule.val_dataloader()))
    #     val_samples = next(iter(trainer.train_dataloader))
    #     val_imgs, val_labels = val_samples
    #     val_imgs = [x.to(device=pl_module.device) for x in val_imgs]

    #     # val_imgs = val_imgs.to(device=pl_module.device)
    #     box_preds = pl_module(val_imgs)

    #     images_to_log = []
    #     for index, (boxes,img) in enumerate(zip(box_preds, val_imgs)):
    #         bbs = boxes["boxes"]
    #         labels = boxes["labels"]
    #         scores = boxes["scores"]
    #         positions = []
    #         # for box, label in zip(bbs, labels):
    #         for i in range(len(labels)):
    #             box = bbs[i]
    #             label = labels[i]
    #             score = scores[i]
    #             box_pos = {
    #                 "position" :{
    #                     "minX" : int(box[0]),
    #                     "minY" : int(box[1]),
    #                     "maxX" : int(box[2]),
    #                     "maxY" : int(box[3])
    #                 },
    #                 "class_id" : int(label),
    #                 "scores" : {"confidence" : float(score)},
    #                 "box_caption" : str(int(label)) + ":" + str(round(float(score), 3)),
    #                 "domain" : "pixel"
    #             }
    #             positions.append(box_pos)
    #         bbs_gt = val_labels[index]["boxes"]
    #         labels_gt = val_labels[index]["labels"]
    #         gt_positions = []
    #         for box, label in zip(bbs_gt, labels_gt):
    #             box_pos = {
    #                 "position" :{
    #                     "minX" : int(box[0]),
    #                     "minY" : int(box[1]),
    #                     "maxX" : int(box[2]),
    #                     "maxY" : int(box[3])
    #                 },
    #                 "class_id" : int(label),
    #                 "domain" : "pixel"
    #             }
    #             gt_positions.append(box_pos)

    #         boxes_data = {
    #             "predictions" : {
    #                 "box_data" : positions
    #             },
    #             "ground_truth" : {
    #                 "box_data" : gt_positions
    #             }

    #         }
    #         wandb_img = wandb.Image(img, boxes=boxes_data)
    #         images_to_log.append(wandb_img)

    #     wandb.log({"training_batch" : images_to_log})


    def on_validation_epoch_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        # val_samples = next(iter(trainer.datamodule.val_dataloader()))
        val_samples = next(iter(trainer.val_dataloaders[0]))
        val_imgs, val_labels = val_samples
        val_imgs = [x.to(device=pl_module.device) for x in val_imgs]

        # val_imgs = val_imgs.to(device=pl_module.device)
        box_preds = pl_module(val_imgs)

        images_to_log = []
        for index, (boxes,img) in enumerate(zip(box_preds, val_imgs)):
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
                    "position" :{
                        "minX" : int(box[0]),
                        "minY" : int(box[1]),
                        "maxX" : int(box[2]),
                        "maxY" : int(box[3])
                    },
                    "class_id" : int(label),
                    "scores" : {"confidence" : float(score)},
                    "box_caption" : str(int(label)) + ":" + str(round(float(score), 3)),
                    "domain" : "pixel"
                }
                positions.append(box_pos)
            bbs_gt = val_labels[index]["boxes"]
            labels_gt = val_labels[index]["labels"]
            gt_positions = []
            for box, label in zip(bbs_gt, labels_gt):
                box_pos = {
                    "position" :{
                        "minX" : int(box[0]),
                        "minY" : int(box[1]),
                        "maxX" : int(box[2]),
                        "maxY" : int(box[3])
                    },
                    "class_id" : int(label),
                    "domain" : "pixel"
                }
                gt_positions.append(box_pos)

            boxes_data = {
                "predictions" : {
                    "box_data" : positions
                },
                "ground_truth" : {
                    "box_data" : gt_positions
                }

            }
            wandb_img = wandb.Image(img, boxes=boxes_data)
            images_to_log.append(wandb_img)

        wandb.log({"validation_batch" : images_to_log})


class CarsDataset(Dataset):
    def __init__(self, img_root, ann_file):
        super().__init__()
        self.root = Path(img_root)
        self.df = pd.read_csv(ann_file, index_col='image')
        # self.ids = list(df.index.unique().values)
        self.ids = [_.name for _ in self.root.iterdir()]
        self.ids_with_box = self.df.index.unique().values
        self.transforms = A.Compose([
            A.Resize(width=480, height=480),
            A.Normalize(mean=(0.2432, 0.3255, 0.3522) , std = (0.2186, 0.2592, 0.3083)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.4, min_area=5))
        self.num_cat = 1
        self.imgs = [Image.open(self.root / p) for p in self.ids]

    def __getitem__(self, index):
        img = self.imgs[index]
        if self.ids[index] in self.ids_with_box:
            target = self.df.loc[self.ids[index]].values
        else:
            target = np.array([])

        if target.ndim == 1:
            target = np.expand_dims(target, axis=0)
        n_boxes = np.size(target, 0)
        if np.size(target, 1) == 0:
            n_boxes = 0
            target = []

        labels = [1 for _ in range(n_boxes)]
        transformed = self.transforms(image=np.array(img), bboxes=target, class_labels=labels)
        target = {}
        img = transformed['image']
        target["boxes"] = torch.tensor(transformed["bboxes"]) if len(transformed["bboxes"]) != 0 else torch.empty((0,4))
        target["labels"] = torch.tensor(transformed["class_labels"]).type(torch.int64)

        return img, target

    
    def __len__(self) -> int:
        return len(self.ids)


class COCODataset(Dataset):
    def __init__(self, root, annFile):
        super().__init__()
        self.root = root 
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.Atransforms = A.Compose([
            A.Resize(width=224, height=224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.4, min_area=5))
        self.category_names = {}
        for cat in self.coco.loadCats(self.coco.getCatIds()):
            self.category_names[cat['id']] = cat['name']
    
    def _load_image(self, id : int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(Path(self.root)/path).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        targets = self.coco.loadAnns(self.coco.getAnnIds(id))
        boxes = []
        labels = []
        for box in targets:
            # if box["bbox"][2] == 0 or box["bbox"][3] == 0:
            #     #TODO change to log warn 
            #     print(f"Warn! bbox with id:{box['id']} is invalid")
            #     continue
            boxes.append(xywh_to_xyxy(box["bbox"]))
            labels.append(box["category_id"])

        return {"boxes" : torch.tensor(boxes) if len(boxes) != 0 else torch.empty((0,4)),
                "labels" : torch.tensor(labels).type(torch.int64) }

    def __getitem__(self, index):
        id = self.ids[index]
        img = self._load_image(id)
        target = self._load_target(id)
        transformed = self.Atransforms(image=np.array(img), bboxes=target["boxes"], class_labels=target["labels"])
        img = transformed['image']
        target["boxes"] = torch.tensor(transformed["bboxes"]) if len(transformed["bboxes"]) != 0 else torch.empty((0,4))
        target["labels"] = torch.tensor(transformed["class_labels"]).type(torch.int64)

        # transformed = self.transforms(img)
        return img, target
    
    def __len__(self) -> int:
        return len(self.ids)

def collate(batch):
    return tuple(zip(*batch))

# val_dataset = COCODataset(val_root, val_annotations)
# train_dataset = COCODataset(train_root, train_annotations)
cars_annotations = "/home.stud/kuntluka/dataset/cars/data/train_solution_bounding_boxes.csv"
training_root = "/home.stud/kuntluka/dataset/cars/data/training_images"
# val_dataset = COCODataset(training_root, cars_annotations)
val_dataset = CarsDataset(training_root, cars_annotations)

train_loader = DataLoader(val_dataset, batch_size=4, collate_fn=collate, num_workers=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, collate_fn=collate, num_workers=8, shuffle=False)
# val_loader = DataLoader(val_dataset, batch_size = 4, collate_fn=collate, num_workers=4, shuffle=False)
# foo = next(iter(val_loader))
model = FasterRCNN(num_classes = 2, pretrained=True)
wandb_logger = WandbLogger()
trainer = Trainer(gpus=1, auto_select_gpus=True, auto_lr_find = True,
            logger=wandb_logger, callbacks=LogImagePredictions(),
            limit_val_batches=2, min_epochs=500)
trainer.tune(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)