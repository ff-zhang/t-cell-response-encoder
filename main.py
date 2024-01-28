import numpy as np
import torch
from torch.utils import data
from sklearn.model_selection import KFold

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
    'max_epochs': 200,
    'df': [i for i in range(1, 7)],
    'save': True,
}


def MAPE_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculates the mean absolute percentage difference between the input and the target.
    """
    return torch.mean(torch.abs((target - input) / target))


def train_model(df: dataset.CytokineDataset, kfold: KFold, filename=None, criterion=None, write=False):
    if write is True and filename is None:
        raise FileNotFoundError

    if write:
        f = open(filename, 'w')
        f.close()

    print(f'Training on dataset {params["df"]} with {len(df)} points ({torch.seed()})\n')

    losses = {}

    for i, (train_index, test_index) in enumerate(kfold.split(df)):
        print(f'---------------- Fold {i + 1} ----------------')

        learn_rates = [0.005, 0.001, 0.0005, 0.0001]
        for lr in learn_rates:
            print(f'\tLearning rate : {lr}')

            train_loader = data.DataLoader(data.Subset(df, train_index), batch_size=2, shuffle=True)
            test_loader = data.DataLoader(data.Subset(df, test_index), batch_size=2, shuffle=True)

            nn = model.CytokineModel(h1=128, h2=16, output=5 * 45)
            if criterion == 'MAPE':
                criterion = MAPE_loss
            else:
                criterion = torch.nn.MSELoss('mean')
            optimizer = torch.optim.Adam(nn.parameters(), lr)

            losses[lr] = model.train(nn, train_loader, test_loader, criterion, optimizer, **params)

        plot.plot_loss(losses, (params['df'], i + 1))

    if write:
        with open(filename, 'a') as f:
            f.write(', '.join(str(w) for w in nn.fc1.weight.detach().numpy().flatten()) + ', ')
            f.write(', '.join(str(w) for w in nn.fc2.weight.detach().numpy().flatten()) + '\n')


if __name__ == '__main__':
    # PyTorch seed for a nice training and testing dataset,
    np.random.seed(0)
    torch.manual_seed(5667615556836105505)

    df = dataset.CytokineDataset([f'PeptideComparison_{i}' for i in range(1, 7)])
    plot.plot_dataset(df, 'IL-17A')
    for f in [f'PeptideComparison_{i}' for i in range(7, 10)]:
        df = dataset.CytokineDataset([f])
        plot.plot_dataset(df, 'IL-17A')

    df, kf = dataset.get_kfold_dataset(params, n_splits=4)
    train_model(df, kf)

    # Load the manually saved trained model which trained using the fixed seed.
    nn = torch.load('model/nn-0.001-[1, 2, 3, 4, 5, 6]/eph-80.pth')

    import matplotlib.pyplot as plt

    nn.eval()
    with torch.no_grad():
        for n in [10, 21, 42, 66]:
            x, y, r = df[n]
            pred = nn(x)
            pred = torch.reshape(pred, (5, 45)).detach().numpy()
            r = r.detach().numpy()
            for i, c in enumerate(['blue', 'green', 'red', 'orange', 'purple']):
                plt.plot(pred.T[:, i], color=c)
                plt.plot(r.T[:, i], alpha=0.5, color=c)

            plt.title(f'Target and Prediction ({n})')
            plt.show()

    print('hello world!')
