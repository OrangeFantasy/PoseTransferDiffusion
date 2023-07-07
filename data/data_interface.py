import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from diffusion.utils import instantiate_from_config


class DataInterFace(pl.LightningDataModule):
    def __init__(self, train_dataset_config, test_dataset_config, train_batch_size, test_batch_size, num_workers: int = 8) -> None:
        super().__init__()
        self.train_dataset_config = train_dataset_config
        self.test_dataset_config = test_dataset_config
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_dataset = None
        self.test_dataset = None
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        pass

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        self.train_dataset = instantiate_from_config(self.train_dataset_config)
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        self.test_dataset = instantiate_from_config(self.test_dataset_config)
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, num_workers=self.num_workers, pin_memory=True)
