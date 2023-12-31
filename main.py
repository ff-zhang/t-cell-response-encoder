import torch
from torch.utils import data

import numpy as np
import matplotlib.pyplot as plt

from scripts import dataset, model, plot

LEVEL_VALUES = [
    # Data
    ['PeptideComparison_1', 'PeptideComparison_2', 'PeptideComparison_3', 'PeptideComparison_4', 'PeptideComparison_5', 'TCellNumber_1', 'Activation_1', 'PeptideComparison_6', 'PeptideComparison_7', 'PeptideComparison_8', 'PeptideComparison_9', 'TCellNumber_2', 'TCellNumber_3', 'Activation_2', 'TCellNumber_4', 'Activation_3'],
    # T cell counts
    ['200k', '100k', '80k', '32k', '30k', '16k', '10k', '8k', '3k', '2k'],
    # Peptides
    ['N4', 'Q4', 'T4', 'V4', 'G4', 'E1'],   # ['A2', 'Y3', 'A8', 'Q7']
    # Concentrations
    ['1uM', '300nM', '100nM', '30nM', '10nM', '3nM', '1nM', '100pM', '10pM'],
    # Time points
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71],
    # Cytokines
    ['IFNg', 'IL-10', 'IL-17A', 'IL-2', 'IL-4', 'IL-6', 'TNFa'],
    # Features
    ['concentration', 'derivative', 'integral']
]

params = {
    'max_epochs': 100,
    'df': 'all',
    'save': True,
}


def MAPE_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculates the mean absolute percentage difference between the input and the target.
    """
    return torch.mean(torch.abs((target - input) / target))


def train_model(datasets: list[data.Subset], filename=None, criterion=None, write=False):
    if write is True and filename is None:
        raise FileNotFoundError

    if write:
        f = open(filename, 'w')
        f.close()

    assert len(datasets) == 3
    train_set, val_set, test_set = datasets

    print(f'Training on dataset {params["df"]} with {len(train_set)} points ({torch.seed()})\n')

    learn_rates = [0.1, 0.01, 0.001]
    for _ in range(1):
        train_loader = data.DataLoader(train_set, batch_size=1, shuffle=True)
        val_loader = data.DataLoader(val_set, batch_size=1, shuffle=True)

        print(f'-------------------- Iter. {_} --------------------')

        losses = {}

        for lr in learn_rates:
            print(f'\tlearning rate : {lr}')
            nn = model.CytokineModel()

            if criterion == 'MAPE':
                criterion = MAPE_loss
            else:
                criterion = torch.nn.MSELoss('mean')
            optimizer = torch.optim.Adam(nn.parameters(), lr)

            losses[lr] = model.train(nn, train_loader, val_loader, criterion, optimizer, **params)

        plot.plot_loss(losses, (params['df'], len(train_set)), **params)

        if write:
            with open(filename, 'a') as f:
                f.write(', '.join(str(w) for w in nn.fc1.weight.detach().numpy().flatten()) + ', ')
                f.write(', '.join(str(w) for w in nn.fc2.weight.detach().numpy().flatten()) + '\n')


if __name__ == '__main__':
    # PyTorch seed for a nice training and testing dataset,
    torch.manual_seed(5667615556836105505)

    df = dataset.CytokineDataset([f'PeptideComparison_{i}' for i in range(1, 7)])
    plot.plot_dataset(df, 'IL-17A')
    for f in [f'PeptideComparison_{i}' for i in range(7, 10)]:
        df = dataset.CytokineDataset([f])
        plot.plot_dataset(df, 'IL-17A')

    dfs = dataset.get_train_test_subset(params)
    train_model(dfs)

    # Load the manually saved trained model which trained using the fixed seed.
    nn = torch.load('model/good-nn-0.01-all/eph-60.pth')
    preds = model.evaluate(nn, data.DataLoader(dfs[0], batch_size=1, shuffle=True))
    plot.plot_pred_concentration(preds, dfs[0])

    print('hello world!')
