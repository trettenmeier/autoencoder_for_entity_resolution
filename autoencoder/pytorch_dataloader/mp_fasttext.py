import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from ..utils.get_resources import get_config
from ..pytorch_datasets.mp_fasttext import MPFasttextDataset


def create_pytorch_dataloader_fasttext_for_autoencoder_training(df, fasttext_model):
    """
    Returns train and val dataloader for training the autoencoder with fasttext embeddings.

    :param df:
    :param fasttext_model:
    :return:
    """
    config = get_config()

    only_matches_dataset = MPFasttextDataset(
        df=df,
        fasttext_model=fasttext_model,
        truncate_vectors_to_x_words=config["fasttext_embedding"]["truncate_vectors_to_x_words"],
        size_embedding_vector=config["fasttext_embedding"]["size_embedding_vector"]
    )

    # Creating data indices for training and validation splits:
    validation_split = 0.1
    shuffle_dataset = config["dataloader"]["shuffle"]
    random_seed = config["dataloader"]["random_seed"]

    dataset_size = len(only_matches_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # creating dataloaders
    train_loader = DataLoader(
        only_matches_dataset,
        batch_size=config["dataloader"]["batch_size"],
        sampler=train_sampler
    )

    val_loader = DataLoader(
        only_matches_dataset,
        batch_size=config["dataloader"]["batch_size"],
        sampler=val_sampler
    )

    return train_loader, val_loader


def create_pytorch_dataloader_train_test(df_train: pd.DataFrame, df_test: pd.DataFrame, fasttext_model):
    """
    Return dataloader for train and test data to be fed through the trained autoencoder as preparation for training
    the downstream classifier.

    :param df_train:
    :param df_test:
    :param fasttext_model:
    :return:
    """
    config = get_config()

    train_dataset = MPFasttextDataset(
        df=df_train,
        fasttext_model=fasttext_model,
        truncate_vectors_to_x_words=config["fasttext_embedding"]["truncate_vectors_to_x_words"],
        size_embedding_vector=config["fasttext_embedding"]["size_embedding_vector"]
    )

    test_dataset = MPFasttextDataset(
        df=df_test,
        fasttext_model=fasttext_model,
        truncate_vectors_to_x_words=config["fasttext_embedding"]["truncate_vectors_to_x_words"],
        size_embedding_vector=config["fasttext_embedding"]["size_embedding_vector"]
    )

    train_loader = DataLoader(train_dataset, batch_size=config["dataloader"]["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["dataloader"]["batch_size"])

    return train_loader, test_loader
