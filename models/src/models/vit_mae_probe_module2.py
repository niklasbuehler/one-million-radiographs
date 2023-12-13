from typing import Any, Dict

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy
from your_project import VisionTransformerMAE  # Update with the correct import path

class FineProbeModel(LightningModule):
    def __init__(
        self,
        num_classes: int,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        mae_checkpoint: str = "path/to/your/mae/checkpoint.ckpt",
    ) -> None:
        super().__init__()

        # Load the pre-trained ViT MAE model
        self.mae_model = VisionTransformerMAE.load_from_checkpoint(mae_checkpoint)

        # Discard the decoder
        self.mae_model.net.decoder = None

        # Freeze the encoder
        for param in self.mae_model.net.encoder.parameters():
            param.requires_grad = False

        # Add a new fully connected layer for classification
        self.classifier = nn.Linear(self.mae_model.net.embed_dim, num_classes)

        self.accuracy = Accuracy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the encoder of the ViT MAE model
        with torch.no_grad():
            embeddings = self.mae_model.net.encoder(x)

        # Forward pass through the classifier
        logits = self.classifier(embeddings)

        return logits

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch['image'], batch['label']
        logits = self.forward(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch['image'], batch['label']
        logits = self.forward(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True)

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch['image'], batch['label']
        logits = self.forward(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        self.log("test/loss", loss, on_epoch=True, prog_bar=True)
        self.log("test/acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {"optimizer": optimizer}
