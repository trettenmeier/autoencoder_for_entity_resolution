import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import numpy as np


def create_pytorch_dataloader_for_neural_net_classifier(X_train, X_test, y_train, y_test):
    loaders = NeuralNetClassifierTrainingDataLoaderFactory(X_train, y_train)
    classifier_train_loader = loaders.get_train_loader()
    classifier_val_loader = loaders.get_val_loader()

    loaders = NeuralNetClassifierTestDataLoaderFactory(X_test, y_test)
    classifier_test_loader = loaders.get_test_loader()

    return classifier_train_loader, classifier_val_loader, classifier_test_loader


class NeuralNetClassifierTrainingDataLoaderFactory:
    def __init__(self, x, y):
        self.dataset = NNTrainingDataset(x, y)
        self.batch_size = 200
        self.validation_split = 0.1
        self.shuffle_dataset = True
        self.random_seed = 66

        # Creating data indices for training and validation splits:
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(self.validation_split * dataset_size))
        if self.shuffle_dataset:
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.valid_sampler = SubsetRandomSampler(val_indices)

    def get_train_loader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler
        )

    def get_val_loader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.valid_sampler
        )


class NeuralNetClassifierTestDataLoaderFactory:
    def __init__(self, x, y):
        self.dataset = NNTestDataset(x, y)
        self.batch_size = 200

    def get_test_loader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
        )


class NNTrainingDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y.values.squeeze()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        return torch.from_numpy(self.X.iloc[idx].to_numpy()), torch.tensor(self.y[idx])


class NNTestDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y.values.squeeze()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X.iloc[idx].to_numpy()), torch.tensor(self.y[idx])

