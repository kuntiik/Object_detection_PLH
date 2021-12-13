from typing import Any, List
import torch
from pytorch_lightning import LightningModule
from src.modules.models.simple_dense_net import SimpleDenseNet
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import MaxMetric

class MNISTModel(LightningModule):
    def __init__(self, input_size : int = 784, hidden_size : int = 256, output_size :int = 10, lr : float = 1e-3, weight_decay : float = 5e-4):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = SimpleDenseNet(hparams=self.hparams)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor()):
        return self.model(x)

    def step(self, batch : Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y
    
    def training_step(self, batch : Any, batch_idx : int):
        loss, preds, targets = self.step(batch)

        self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=False)

        return {"loss" : loss, "preds" : preds, "targets" : targets}
    
    def validation_step(self, batch : Any, batch_idx : int):
        loss, preds, targets = self.step(batch) 

        self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss" : loss, "preds" : preds, "targets" : targets}
        
    def validation_epoch_end(self, outputs):
        acc = self.val_acc.compute()
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best, on_epoch=True, prog_bar=True)
        self.val_acc.reset()
    
    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)

        self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)

        return {"loss" : loss, "preds" : preds, "targets" : targets}
    
    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
