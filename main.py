import os
import glob
from copy import deepcopy

import torch
from torch.utils import data

import matplotlib.pyplot as plt

from scripts import dataset, model

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


def plot_loss(losses):
    for lr, loss in losses.items():
        train_loss, val_loss, _ = losses[lr]
        plt.plot(train_loss, label=str(lr))

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def main():
    dfs = {'all': dataset.CytokineDataset(
            [f for f in glob.glob(os.path.join('data/final', '*PeptideComparison*.pkl'))]
        )}
    for i in range(1, 10):
        dfs[i] = dataset.CytokineDataset(
            [f for f in glob.glob(os.path.join('data/final', f'*PeptideComparison_{i}*.pkl'))]
        )

    train_percent = 0.7
    valid_percent = 0.15
    test_percent = 0.15

    for n, df in dfs.items():
        train_num = int(train_percent * len(df))
        valid_num = int(valid_percent * len(df))
        test_num = len(df) - (train_num + valid_num)

        train_set, val_set, test_set = data.random_split(df, [train_num, valid_num, test_num])

        train_loader = data.DataLoader(train_set, batch_size=1, shuffle=True)
        val_loader = data.DataLoader(val_set, batch_size=1, shuffle=True)
        test_loader = data.DataLoader(test_set, batch_size=1, shuffle=True)

        params = {
            'max_epochs': 100
        }

        losses = {lr: None for lr in [0.1, 0.01, 0.05, 0.001, 0.0005, 0.0001]}
        for lr in losses.keys():
            nn = model.CytokineModel(6, 2, 5)

            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(nn.parameters(), lr)

            train_loss, val_loss = model.train(nn, train_loader, val_loader, criterion, optimizer, **params)
            losses[lr] = [train_loss, val_loss, deepcopy(nn)]
            torch.save(nn, f'model/nn-{n}-{lr}.pth')

        plot_loss(losses)


if __name__ == '__main__':
    main()

    for f in glob.glob('model/*.pth'):
        model = torch.load(f)
        f = f[: -4].split('-')
        print(f[-1] + ':')
        for parameter in model.parameters():
            print(parameter.data)

        print()

    print('hello world!')
