import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS

from diffusion.utils import instantiate_from_config


class DataInterFace(pl.LightningDataModule):
    def __init__(self, dataset_config, train_batch_size, test_batch_size, num_workers: int = 8) -> None:
        super().__init__()
        self.dataset_config = dataset_config
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        self.dataset = None
    
    def setup(self, stage: str) -> None:
        # if stage == "fit":
        #     self.dataset = DeepFashionDataset()
        self.dataset = instantiate_from_config(self.dataset_config)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.dataset, batch_size=self.train_batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)
