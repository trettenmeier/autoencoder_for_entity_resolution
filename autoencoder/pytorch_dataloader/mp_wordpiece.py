import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from ..pytorch_datasets.mp_wordpiece import MPWordpieceDataset, MPWordpieceDatasetFixedPositions

from ..utils.get_resources import get_config


def create_pytorch_dataloader_wordpiece_for_autoencoder_training(df, wordpiece_tokenizer):
    config = get_config()

    batch_size = config["dataloader"]["batch_size"]
    validation_split = 0.1
    shuffle_dataset = config["dataloader"]["shuffle"]
    random_seed = config["dataloader"]["random_seed"]

    if config["wordpiece_tokenizer"]["fixed_position_of_values"]:
        dataset = MPWordpieceDatasetFixedPositions(
            df,
            wordpiece_tokenizer,
            config["wordpiece_tokenizer"]["number_of_tokens_per_datapoint"]
        )
    else:
        dataset = MPWordpieceDataset(
            df,
            wordpiece_tokenizer,
            config["wordpiece_tokenizer"]["number_of_tokens_per_datapoint"]
        )

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader


def create_pytorch_dataloader_train_test(df_train: pd.DataFrame, df_test: pd.DataFrame, wordpiece_tokenizer):
    config = get_config()

    if config["wordpiece_tokenizer"]["fixed_position_of_values"]:
        train_dataset = MPWordpieceDatasetFixedPositions(
            df_train, wordpiece_tokenizer, config["wordpiece_tokenizer"]["number_of_tokens_per_datapoint"])
        test_dataset = MPWordpieceDatasetFixedPositions(
            df_test, wordpiece_tokenizer, config["wordpiece_tokenizer"]["number_of_tokens_per_datapoint"])
    else:
        train_dataset = MPWordpieceDataset(
            df_train, wordpiece_tokenizer, config["wordpiece_tokenizer"]["number_of_tokens_per_datapoint"])
        test_dataset = MPWordpieceDataset(
            df_test, wordpiece_tokenizer, config["wordpiece_tokenizer"]["number_of_tokens_per_datapoint"])

    train_loader = DataLoader(train_dataset, batch_size=config["dataloader"]["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["dataloader"]["batch_size"])

    return train_loader, test_loader
