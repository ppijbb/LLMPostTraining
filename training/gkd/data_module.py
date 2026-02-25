from typing import Iterable, Optional, Callable, Any
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

class DistilDataset(Dataset):
    def __init__(
        self, 
        data: Iterable, 
        transform: Optional[Callable] = None
    ):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self,
        idx:int
    )->Any:
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class DistillationDataModule:
    def __init__(
        self,
        train_data: Iterable,
        val_data: Iterable,
        test_data: Iterable,
        processor: ProcessorMinxin,
        batch_size: int = 32,
        num_workers: int = -1,
        pin_memory: bool = True,
        drop_last: bool = True,
    ):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def transform(self):
        def transform_fn(sample: Any):
            return self.processor.apply_chat_template(sample, return_tensors="pt")
        return transform_fn

    def setup(
        self,
        stage=None
    ):
        self.train_dataset = DistilDataset(self.train_data, transform=self.transform)
        self.val_dataset = DistilDataset(self.val_data, transform=self.transform)
        self.test_dataset = DistilDataset(self.test_data, transform=self.transform)

    def _get_data_loader(
        self,
        dataset: Dataset
    )->DataLoader:
        return DataLoader(
            dataset=dataset, 
            batch_size=self.batch_size, 
            shuffle=(dataset==self.train_dataset),
            num_workers=self.num_workers if dataset==self.train_dataset else 1)

    def train_dataloader(self):
        return self._get_data_loader(dataset=self.train_dataset)

    def val_dataloader(self):
        return self._get_data_loader(dataset=self.val_dataset)

    def test_dataloader(self):
        return self._get_data_loader(dataset=self.test_dataset)

def process_samples(sample: Any, processor: ProcessorMixin):
    return processor.apply_chat_template(sample, return_tensors="pt")