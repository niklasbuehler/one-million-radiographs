from transformers import ViTImageProcessor
from typing import Tuple, Dict, Any

import torch
from lightning import LightningModule
import torch.nn as nn
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import MaxMetric, MeanMetric
from src.models.vit_mae_module import VisionTransformerMAE

class MAEFineProbeBodyPartClassifier(LightningModule):
    def __init__(self, num_classes: int, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, mae_checkpoint: str, seq_mean: bool, compile: bool):
        super().__init__()

        self.save_hyperparameters(logger=False)

        # Load the pre-trained ViT MAE model
        print("checkpoint path", mae_checkpoint)
        self.mae_model = VisionTransformerMAE.load_from_checkpoint(mae_checkpoint)
        # Disable masking
        self.mae_model.net.config.mask_ratio = 0
        self.image_processor = self.mae_model.image_processor

        # Discard the decoder
        self.mae_model.net.decoder = None

        # Freeze the encoder
        for param in self.mae_model.net.vit.encoder.parameters():
            param.requires_grad = False

        # Add a new fully connected layer for classification
        self.seq_mean = seq_mean
        hidden_size = self.mae_model.net.config.hidden_size # 768
        #num_patches = 50 # with masking! is _not_ equal to self.mae_model.net.vit.embeddings.num_patches
        num_patches = self.mae_model.net.vit.embeddings.num_patches # 197
        last_dim = hidden_size * (num_patches+1) # +1 for the mask token
        if self.seq_mean:
            last_dim = hidden_size
        self.classifier = nn.Linear(last_dim, num_classes)

        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the encoder of the ViT MAE model
        with torch.no_grad():
            inputs = self.image_processor(x, return_tensors="pt").to(self.device) # BatchFeature
            latent = self.mae_model.net.vit(**inputs).last_hidden_state

        #print("latent:", latent.size()) # [32, 50, 768]
        # sequence length is regarding patches?
        # if yes, it makes more sense to flatten
        if self.seq_mean:
            embeddings = latent.mean(dim=1)
        else:
            embeddings = latent.flatten(start_dim=1)
        #print("embeddings:", embeddings.size()) # mean: [32, 768], flatten: [32, 38400], flatten+nomask: [128, 151296]

        # Forward pass through the classifier
        logits = self.classifier(embeddings)

        return logits

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        #print("x:", x.size()) # [32, 3, 384, 384]
        logits = self.forward(x)
        #print("logits:", logits.size()) # [32, 9]
        #print("y:", y.size()) # [32]
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.classifier.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
