import os

import pytest
import torch

from src.datamodules.tse_datamodule import TSEDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_tse_datamodule(batch_size):
    datamodule = TSEDataModule(batch_size=batch_size)
    datamodule.prepare_data()

    assert not datamodule.data_train and not datamodule.data_val

    assert os.path.exists(os.path.join("data", "tweet-sentiment-extraction"))

    datamodule.setup()

    assert datamodule.data_train and datamodule.data_val
    assert datamodule.train_dataloader()
    assert datamodule.val_dataloader()

    batch = next(iter(datamodule.train_dataloader()))
