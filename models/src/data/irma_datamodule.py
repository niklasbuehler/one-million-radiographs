from typing import Any, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from src.data.components.irma_dataset import IRMADataset
from src.data.components.irma_util import Irma

class IRMADataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "../data/IRMA",
            train_val_test_split: Tuple[int, int, int] = (8_000, 2_000, 2_677),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            image_size: int = 224, # resnet50: 224, ViT: 384
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_dir = data_dir
        self.train_val_test_split = train_val_test_split
        self.batch_size_per_device = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_size = image_size

        self.data_train: Optional[torch.utils.data.Dataset] = None
        self.data_val: Optional[torch.utils.data.Dataset] = None
        self.data_test: Optional[torch.utils.data.Dataset] = None

    def prepare_data(self) -> None:
        irma_dataset = Irma(self.data_dir)
        irma_dataset.load()

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.data_train and not self.data_val and not self.data_test:
            irma_dataset = Irma(self.data_dir)
            irma_dataset.load()

            # Create an instance of the custom IrmaDataset class
            custom_dataset = IRMADataset(df=irma_dataset.df, irma_util=irma_dataset, image_size=self.image_size)

            # Split the dataset
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=custom_dataset,
                lengths=self.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass


if __name__ == "__main__":
    _ = IRMADataModule()
