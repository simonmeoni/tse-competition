from typing import Optional

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import BatchSampler, DataLoader, Dataset, Subset, SubsetRandomSampler
from transformers import AutoTokenizer


class TSEDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset.iloc[idx]
        return example.to_dict()


class TSEDataModule(LightningDataModule):
    """
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/tweet-sentiment-extraction/train.csv",
        train_val_test_split: float = 0.8,
        batch_size: int = 32,
        val_batch_size: int = 32,
        num_workers: int = 0,
        max_length: int = 256,
        pin_memory: bool = False,
        tokenizer: str = "roberta-base",
        k_folds: int = 5,
        current_fold: int = 0,
    ):
        super().__init__()

        self.current_fold = current_fold
        self.k_folds = k_folds
        self.data_dir = data_dir
        self.train_val_split = train_val_test_split
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length
        self.data_train: Optional[Subset] = None
        self.data_val: Optional[Subset] = None
        self.full_dataset = None

    def prepare_data(self):
        csv = pd.read_csv(self.data_dir)
        csv = csv[csv["sentiment"] != "neutral"]
        csv["start_positions"] = ""
        csv["end_positions"] = ""

        for index, example in csv.iterrows():
            context = example["text"].strip()
            response = example["selected_text"].strip()
            question = example["sentiment"] + "?"
            encodings = self.tokenizer(
                question,
                context,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            start_idx = context.find(response)
            start_token = encodings.char_to_token(0, start_idx, 1)
            end_idx = context.find(response) + len(response) - 1
            end_token = encodings.char_to_token(0, end_idx, 1)
            csv.at[index, "question"] = question
            csv.at[index, "start_positions"] = torch.LongTensor([start_token])
            csv.at[index, "end_positions"] = torch.LongTensor([end_token])
        self.full_dataset = TSEDataset(csv)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val."""
        stratified_k_fold = StratifiedKFold(n_splits=self.k_folds, shuffle=True)
        data_train_ids, data_val_ids = list(
            stratified_k_fold.split(
                self.full_dataset,
                [
                    1 if sentiment == "positive?" else 0
                    for sentiment in self.full_dataset.dataset["question"]
                ],
            )
        )[self.current_fold]
        self.data_train = Subset(self.full_dataset, data_train_ids)
        self.data_val = Subset(self.full_dataset, data_val_ids)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        collate = torch.utils.data.dataloader.default_collate(batch)
        encodings = self.tokenizer(
            collate["question"],
            collate["text"],
            truncation=True,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
        )
        encodings.update(
            {
                "start_positions": collate["start_positions"],
                "end_positions": collate["end_positions"],
            }
        )
        return encodings, collate["selected_text"]
