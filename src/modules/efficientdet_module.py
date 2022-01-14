from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.parsing import save_hyperparameters
from src.modules.models.efficientdet import create_model
# from models.efficientdet import create_model
import torch
from torchmetrics import MAP



class EfficientDetModule(LightningModule):

    def __init__(self, 
        num_classes=1,
        img_size=512,
        prediction_confidence_threshold=0.2,
        learning_rate=0.0002,
        model_architecture='tf_efficientnetv2_m'
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = create_model(num_classes, img_size, model_architecture)
        self.val_mAP = MAP()

    @staticmethod
    def from_effdet_preds(preds, labels):
        predictions = []
        for pred in preds:
            det = pred[:, [1,0,3,2]]
            sc = pred[:, 4]
            lab = pred[:, 5]
            p = dict(
                boxes = det,
                scores = sc,
                labels=lab
            )
            predictions.append(p)

        targets = []
        for label, box in zip(labels['cls'],labels['bbox']):
            targets.append(
                dict(
                    boxes = box[:, [1,0,3,2]],
                    labels = label
                )
            )

        return predictions, targets
        


    def forward(self, images, targets):
        return self.model(images, targets)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        losses = self.model(x, y)

        self.log_dict(losses, on_step=True)
        return losses['loss']
        
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x, y)
        preds = outputs['detections']
        predictions, targets = self.from_effdet_preds(preds, y)
        self.val_mAP(predictions, targets)
    
    def validation_epoch_end(self, outputs):
        map_dict = self.val_mAP.compute()
        self.val_mAP.reset()
        self.log_dict(map_dict)
