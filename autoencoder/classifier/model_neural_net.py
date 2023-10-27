import torch.nn as nn
import torch.nn.functional as F


def neural_net_classifier_model(size_latent):
    return NeuralNetClassifier(len_input=2 * size_latent)


class NeuralNetClassifier(nn.Module):
    def __init__(self, len_input):
        super().__init__()
        self.fc1 = nn.Linear(len_input, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 2)

        self.bn1 = nn.BatchNorm1d(len_input)
        self.bn2 = nn.BatchNorm1d(500)
        self.bn3 = nn.BatchNorm1d(200)

        # self.dropout1 = nn.Dropout(0.5)
        # self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.bn1(x)

        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        # x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.bn3(x)
        # x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        return x
