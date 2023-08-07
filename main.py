import os
import glob

import torch
from torch.utils import data

import pandas as pd
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


def plot_loss(losses, title, **kwargs):
    plt.figure(figsize=(9.6, 4.8))

    for lr, loss in losses.items():
        train_loss, val_loss = loss
        plt.plot(train_loss, label='train: ' + str(lr))
        plt.plot(val_loss, label='val.: ' + str(lr))

    plt.title(f'Dataset(s) {title[0]} - {title[1]} Points')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(f'figure/nn-{kwargs["df"]}.png')


def get_actual_conc(df: dataset.CytokineDataset, norm_y: float) -> float:
    if df.normalize is None:
        raise AttributeError

    if df.normalize == 'min-max':
        return (norm_y * (df.x2 - df.x1) + df.x1)[-11]

    elif df.normalize == 'std-score':
        raise NotImplementedError


def train_model(filename=None, write=False):
    if write is True and filename is None:
        raise FileExistsError

    train_percent, valid_percent, test_percent = 0.7, 0.15, 0.15
    learn_rates = [0.01, 0.001, 0.0001]     # [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

    # for n in datasets:
    params = {
        'max_epochs': 50,
        'df': 'all'
    }

    # antigens = { 'N4': 0, 'T4': 1, 'E1': 2 }
    df = [f'PeptideComparison_{i}' for i in range(1, 10)] if params['df'] == 'all' else [f'PeptideComparison_{params["df"]}']
    df = dataset.CytokineDataset(df, normalize='min-max')

    train_num = int(train_percent * len(df))
    valid_num = int(valid_percent * len(df))
    test_num = len(df) - (train_num + valid_num)

    if write:
        f = open(filename, 'w')
        f.close()

    # input weight matrix: [ a_{11}, a_{12}, a_{13}, a_{14}, a_{15}, a_{16}
    #                        a_{21}, a_{22}, a_{23}, a_{24}, a_{25}, a_{26} ]
    # output weight matrix: [ b_{11}, b_{12}
    #                         b_{21}, b_{22}
    #                         b_{31}, b_{32}
    #                         b_{41}, b_{42}
    #                         b_{51}, b_{52} ]

    train_set, val_set, test_set = data.random_split(df, [train_num, valid_num, test_num])

    for _ in range(50):
        train_loader = data.DataLoader(train_set, batch_size=1, shuffle=True)
        val_loader = data.DataLoader(val_set, batch_size=1, shuffle=True)
        test_loader = data.DataLoader(test_set, batch_size=1, shuffle=True)

        print(f'Training on dataset {params["df"]} with {train_num} points')

        losses = {}

        # TODO: return to looping over learning rates later
        for lr in learn_rates:
            nn = model.CytokineModel(6, 2, 5)

            criterion = torch.nn.MSELoss('mean')
            optimizer = torch.optim.Adam(nn.parameters(), lr)

            train_loss, val_loss = model.train(nn, train_loader, val_loader, criterion, optimizer, **params)
            losses[lr] = [train_loss, val_loss]
            # torch.save(nn, f'model/nn-{params["df"]}-{lr}.pth')

        plot_loss(losses, (params['df'], train_num), **params)

        if write:
            with open(filename, 'a') as f:
                f.write(', '.join(str(w) for w in nn.fc1.weight.detach().numpy().flatten()) + ', ')
                f.write(', '.join(str(w) for w in nn.fc2.weight.detach().numpy().flatten()) + '\n')


def plot_weights(weights: str = 'out/weights.csv', file_format: str = 'pdf'):
    df = pd.read_csv(weights)
    X = list(range(1, df.shape[1] + 1))

    # scatter plot showing average
    plt.figure(figsize=(9.6, 4.8))
    for row in df.iterrows():
        plt.scatter(X, row[1], alpha=0.7, s=8)

    plt.scatter(X, df.sum() / 22, color='black')

    plt.title('Learned Weights on Dataset 1')
    plt.xlabel('Weight')
    plt.ylabel('Value')

    plt.xticks(X)

    # plt.show()
    plt.savefig(f'figure/weights_all_scatter.{file_format}')

    # box plot showing median, first and third quartile, interquartile, outliers
    plt.figure(figsize=(9.6, 4.8))
    plt.boxplot(df)

    plt.title('Learned Weights on Dataset 1')
    plt.xlabel('Weight')
    plt.ylabel('Value')

    # plt.show()
    plt.savefig(f'figure/weights_all_box.{file_format}')

    # error plot showing standard deviations
    plt.figure(figsize=(9.6, 4.8))
    avg = df.mean(0)
    plt.errorbar(X, avg, [df.mean(0) - df.min(0), df.max(0) - avg], fmt='.k', ecolor='gray', lw=1, capsize=2)
    plt.errorbar(X, avg, df.std(0), fmt='ok', lw=3, capsize=4)

    plt.title('Learned Weights on Dataset 1')
    plt.xlabel('Weight')
    plt.ylabel('Value')

    # plt.show()
    plt.savefig(f'figure/weights_all_error.{file_format}')


def model_predictions(file: str = 'model/nn-1-0.001.pth'):
    model = torch.load(file)
    model.eval()

    f = open('out/predictions.csv', 'w')
    print('target, prediction', file=f)

    fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(), sharex=True, sharey=True)

    for n in range(2, 10):
        df = [f for f in glob.glob(os.path.join('data/final', f'*PeptideComparison_{n}*.pkl'))]
        df = dataset.CytokineDataset(df)
        df = data.DataLoader(df, batch_size=1)

        for x, y, r in df:
            target = r[0, :, -2].detach().numpy()
            pred = model(x).squeeze().detach().numpy()

            print(f'{target}, {pred}', file=f)

            axes[n-2].plot(target, label="Target")
            axes[n-2].plot(pred, label="Prediction")
            # axes[n-2].legend()

    plt.show()

    f.close()


if __name__ == '__main__':
    train_model(filename='out/weights_all.csv', write=False)

    # plot_weights('out/weights_all.csv', file_format='png')

    # model_predictions()

    print('hello world!')
