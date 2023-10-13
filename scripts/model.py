import time
from tqdm import tqdm

import numpy as np
import torch
from torch import nn


class CytokineModel(nn.Module):
    def __init__(self, input: int, hidden: int, output: int) -> None:
        super(CytokineModel, self).__init__()

        self.fc1 = nn.Linear(input, hidden)
        self.ac1 = nn.Sigmoid()

        self.fc2 = nn.Linear(hidden, output)
        self.ac2 = nn.ReLU()

        self.main = nn.Sequential(
            self.fc1, self.ac1, self.fc2, self.ac2
        )

    def forward(self, x):
        out = self.main(x)
        return out


def train(model: nn.Module, train_loader, validation_loader, criterion, optimizer, **kwargs):
    train_history, val_history = np.zeros(kwargs['max_epochs']), np.zeros(kwargs['max_epochs'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    s = time.time()
    for epoch in tqdm(range(kwargs['max_epochs'])):
        train_loss, val_loss = 0, 0

        for x, _, r in train_loader:
            optimizer.zero_grad()

            outputs = model(x.to(device))
            # gets the entry at 72 hours
            loss = criterion(outputs.squeeze(), r[:, :, -11].to(device))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        with torch.no_grad():
            for x, _, r in validation_loader:
                outputs = model(x.to(device))
                loss = criterion(outputs.squeeze(), r[:, :, -11].to(device))
                val_loss += loss.item()

        train_history[epoch] = train_loss / len(train_loader.dataset)
        val_history[epoch] = val_loss / len(validation_loader.dataset)

    e = time.time()

    print(f'\tTraining time: {round(e - s, 5)} seconds\n')

    return train_history, val_history
