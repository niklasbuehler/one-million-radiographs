from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import MeanSquaredError
from transformers import ViTImageProcessor, ViTMAEForPreTraining

class VisionTransformerMAE(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        # Load ViT image processor
        self.image_processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-base", do_rescale=False)

        # Load ViT Masked Autoencoder model
        self.net = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

        self.mse_loss = MeanSquaredError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs = self.image_processor(x, return_tensors="pt").to(self.device)
        return self.net(**inputs).logits

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, _ = batch
        #print("Shape of x:", x.size()) # [16, 3, 384, 384]
        #inputs = self.image_processor(x, return_tensors="pt").to(self.device).data["pixel_values"]
        #print(inputs)
        #print(type(inputs))
        #print("Shape of inputs:", inputs["pixel_values"].size()) # [16, 3, 224, 224]

        outputs = self(x)
        #print("Shape of outputs:", outputs.size()) # [16, 196, 768]
        reconstructions = self.net.unpatchify(outputs).to(self.device)
        #print("Shape of reconstructions:", reconstructions.size()) # [16, 3, 224, 224]

        #print("Type of reconstructions:", type(reconstructions))
        #print("Type of inputs:", type(inputs))

        loss = self.mse_loss(reconstructions, inputs)
        return loss, reconstructions

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, _ = self.model_step(batch)

        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, _ = self.model_step(batch)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, _ = self.model_step(batch)

        self.log("test/loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.net.parameters())
        return {"optimizer": optimizer}
