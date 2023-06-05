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


def train(model: nn.Module, train_loader, validation_loader, criterion, optimizer, **args):
    train_history, val_history = np.zeros(args['max_epochs']), np.zeros(args['max_epochs'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    s = time.time()
    for epoch in tqdm(range(args['max_epochs'])):
        train_loss, val_loss = 0, 0
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_loader
            else:
                dataloader = validation_loader

            for x, y, r in dataloader:
                optimizer.zero_grad()

                outputs = model(x.to(device))
                loss = criterion(outputs.squeeze(), r[:, :, -2].to(device))
                loss.backward()
                optimizer.step()

                if phase == 'train':
                    train_loss += loss.item()
                else:
                    val_loss += loss.item()

        train_history[epoch] = train_loss
        val_history[epoch] = val_loss

    e = time.time()

    print(f'\nTraining time: {round(e - s, 5)} seconds\n')

    return train_history, val_history
