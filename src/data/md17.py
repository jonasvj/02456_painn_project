# src/data/md17_datamodule.py

from pytorch_lightning import LightningDataModule
from torch_geometric.datasets import MD17
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import torch


class MD17DataModule(LightningDataModule):
    def __init__(
        self,
        molecule_name: str = "ethanol",
        data_dir: str = "data_md17/",
        batch_size_train: int = 32,
        batch_size_inference: int = 128,
        num_workers: int = 0,
        splits=(1000, 200, 200),
        seed: int = 0,
        subset_size: int | None = None,
    ):
        super().__init__()
        self.molecule_name = molecule_name
        self.data_dir = data_dir
        self.batch_size_train = batch_size_train
        self.batch_size_inference = batch_size_inference
        self.num_workers = num_workers
        self.splits = splits
        self.seed = seed
        self.subset_size = subset_size

        # Estos atributos se rellenarán en setup()
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        """Descarga/prepara el dataset si no está ya en disco."""
        MD17(root=self.data_dir, name=self.molecule_name)

    def setup(self, stage=None):
        """Crea los splits de train/val/test."""
        full_dataset = MD17(root=self.data_dir, name=self.molecule_name)

        # Opcional: usar solo un subconjunto para pruebas rápidas
        if self.subset_size is not None:
            full_dataset = full_dataset[: self.subset_size]

        n_train, n_val, n_test = self.splits
        total = len(full_dataset)

        if n_train + n_val + n_test > total:
            raise ValueError(
                f"Splits {self.splits} suman más que el tamaño del dataset ({total})."
            )

        generator = torch.Generator().manual_seed(self.seed)
        self.data_train, self.data_val, self.data_test = random_split(
            full_dataset, [n_train, n_val, n_test], generator=generator
        )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size_train,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size_inference,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size_inference,
            shuffle=False,
            num_workers=self.num_workers,
        )
