import torch
from pytorch_lightning import LightningModule

from torchvision.models.detection.faster_rcnn import FasterRCNN as torchvision_FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn
from torchmetrics import MAP

# from pl_bolts.models.detection.faster_rcnn import create_fasterrcnn_backbone

class FasterRCNN(LightningModule):
    def __init__(self, learning_rate : float = 0.0001, num_classes : int = 1, exclude_bn_bias : bool = True,
            pretrained : bool = False, pretrained_backbone : bool = False, trainable_backbone_layers : int = 5, weight_decay : float = 0.005):
        super().__init__()

        self.save_hyperparameters()
        self.model = fasterrcnn_resnet50_fpn(
            pretrained=self.hparams.pretrained,
            pretrained_backbone=self.hparams.pretrained_backbone,
            trainable_backbone_layers=self.hparams.trainable_backbone_layers
        )

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.hparams.num_classes + 1)
        self.val_mAP = MAP()
        self.train_mAP = MAP()
        self.mAP_keys = ["map_small", "map_medium", "map_large", "map", "map_50", "map_75"]

    def forward(self, x):
        self.model.eval()
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        # fasterrcnn takes both images and targets for training, returns
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log_dict(loss_dict, on_step=False, on_epoch=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "log": loss_dict}
    

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # fasterrcnn takes only images for eval() mode
        outs = self.model(images)
        self.val_mAP(outs, targets)

    def validation_epoch_end(self, outputs):
        map_dict = self.val_mAP.compute()
        self.val_mAP.reset()
        log_map_dict = {}
        for key, value in map_dict.items():
            if key in self.mAP_keys:
                log_map_dict["val/" + key] = value
        self.log_dict(log_map_dict)

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=("bias", "bn")):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {
                "params": excluded_params,
                "weight_decay": 0.0,
            },
        ]

    def configure_optimizers(self):
        if self.hparams.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.hparams.weight_decay)
        else:
            params = self.parameters()

        return torch.optim.Adam(
            params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        # return torch.optim.ADAM(
            

        # )

    