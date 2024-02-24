import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.utils import data

TIME_SERIES_LENGTH = 64
POINT_INDEX = -16


class CytokineModel(nn.Module):
    """
    Model for the internal process of a T-cell using two hidden layers.
    """
    def __init__(self, input: int = 6, h1: int = 8, h2: int = 5, pred: str = 'point'):
        super(CytokineModel, self).__init__()

        assert pred in ['point', 'series']
        self.pred = pred

        self.fc1a = nn.Linear(input, h1)
        self.fc1b = nn.Linear(h1, h1)
        self.fc1c = nn.Linear(h1, h1)
        self.fc1d = nn.Linear(h1, 2)

        self.fc2a = nn.Linear(2, h2)
        # The length of the time series to predict is hard-coded here.
        self.fc2b = nn.Linear(h2, 5 if pred == 'point' else 5 * TIME_SERIES_LENGTH)

        # GeLU
        self.ac1 = nn.Softplus()
        # nn.softplus
        self.ac2 = nn.Softplus()

        self.main = nn.Sequential(
            self.fc1a, self.ac1, self.fc1b, self.ac1, self.fc1c, self.ac1, self.fc1d,
            self.ac1, self.fc2a, self.ac1, self.fc2b
        )

    def forward(self, x):
        out = self.main(x)
        return out


def train(model: nn.Module, train_loader, validation_loader, criterion, optimizer, **kwargs):
    train_history, val_history = np.zeros(kwargs['max_epochs']), np.zeros(kwargs['max_epochs'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    s = time.time()
    model.train()
    for epoch in tqdm(range(kwargs['max_epochs'])):
        train_loss, val_loss = 0, 0

        # Note that only the entry at 64.0 hours is given here.
        for x, _, r in train_loader:
            if model.pred == 'point':
                r = r[:, :, POINT_INDEX]
            elif model.pred == 'series':
                r = torch.reshape(r, (-1, r.shape[1] * r.shape[2]))

            optimizer.zero_grad()

            outputs = model(x.to(device))
            loss = criterion(outputs.squeeze(), r.to(device))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        with torch.no_grad():
            for x, _, r in validation_loader:
                if model.pred == 'point':
                    r = r[:, :, POINT_INDEX]
                elif model.pred == 'series':
                    r = torch.reshape(r, (-1, r.shape[1] * r.shape[2]))

                outputs = model(x.to(device))
                loss = criterion(outputs.squeeze(), r.to(device))
                val_loss += loss.item()

        train_history[epoch] = train_loss / len(train_loader.dataset)
        val_history[epoch] = val_loss / len(validation_loader.dataset)

        if kwargs['save'] and not (epoch + 1) % 5:
            # Only tested when using the Adam optimizer.
            path = Path(f'model/{model.pred}/nn-{optimizer.param_groups[0]["lr"]}-{kwargs["df"]}')
            if not path.exists():
                Path.mkdir(path, exist_ok=True)
            torch.save(model, path / f'eph-{epoch + 1}.pth')

    e = time.time()
    print(f'\tTraining time: {round(e - s, 5)} seconds\n')

    return train_history, val_history


def evaluate(model: nn.Module, loader: data.DataLoader) -> np.array:
    pred = []
    model.eval()
    with torch.no_grad():
        for x, y, r in loader:
            pred.append(model(x).squeeze().detach().numpy())

    return np.array(pred)
