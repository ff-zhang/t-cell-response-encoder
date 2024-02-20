import numpy as np
import pandas as pd
from torch.utils import data

import matplotlib.pyplot as plt

from scripts import dataset

ANTIGENS = ['N4', 'Q4', 'T4', 'V4', 'G4', 'E1']
ANTIGEN_COLOUR = {
    'N4': 'blue',
    'Q4': 'orange',
    'T4': 'green',
    'V4': 'red',
    'G4': 'purple',
    'E1': 'brown',
}
CONCENTRATION_LINE = {
    '100pM': '',
    '10pM': '',
    '300nM': '',
    '100nM': ':',
    '30nM': '',
    '10nM': '-.',
    '3nM': '',
    '1nM': '--',
    '1uM': '-',
}


def plot_spline_process_steps(chosen, df_raw, df_log, df_smooth, df_spline):
    # Extract a few useful variables
    cond, cytokine = chosen
    fig, axes = plt.subplots(2, 2, sharex=True)
    spline = df_spline.loc[cond, cytokine]

    # Time axes. Add (0, 0) to the rescaled data
    exp_times = np.array(df_raw.columns.get_level_values('Time').unique())
    exp_times0 = np.concatenate(([0], exp_times))
    spline_times = np.linspace(exp_times0[0], exp_times0[-1], 201)
    spline_knots = spline.get_knots()

    # y values for each curve
    yraw = df_raw.loc[cond, cytokine]
    ylog = np.concatenate(([0], df_log.loc[cond, cytokine]))
    ysmooth = np.concatenate(([0], df_smooth.loc[cond, cytokine]))
    yspline = spline(spline_times)
    yknots = spline(spline_knots)

    # Curve styles
    style_raw = dict(ls='--', color='k', marker='o', ms=5, lw=2.)
    style_spline = dict(color='xkcd:royal blue', ls='-', lw=3.)
    style_knots = dict(ls='none', marker='^', ms=5, mfc='r', mec='k', mew=1.)
    style_smooth = dict(color='grey', ls=':', marker='s', ms=5, lw=2.)

    # Plot raw data
    ax = axes[0, 0]
    ax.plot(exp_times, yraw, **style_raw, label='Raw')
    ax.set_ylabel(cytokine + ' [nM]')
    ax.set_title('A', loc='left')
    ax.legend()

    # Plot rescaled + smoothing + spline
    ax = axes[1, 1]
    ax.plot(exp_times0, ylog, **style_raw, label='Log')
    ax.plot(exp_times0, ysmooth, **style_smooth, label='Smoothed')
    ax.plot(spline_times, yspline, **style_spline, label='Cubic spline')
    ax.plot(spline_knots, yknots, **style_knots, label='Spline knots')
    ax.set_xlabel('Time (h)')
    ylbl = r'$\log_{10}($' + cytokine + r'$ / \mathrm{LOD})$'
    ax.set_ylabel(ylbl)
    ax.legend()
    ax.set_title('D', loc='left')
    # Use the same limits for the remaining two plots
    ylims = ax.get_ylim()

    # Rescaled data
    ax = axes[0, 1]
    ax.plot(exp_times0, ylog, **style_raw, label='Log')
    ax.set_title('B', loc='left')
    # Same axes for all three plots after rescaling
    ax.set_ylim(ylims)
    ax.legend()
    ax.set_ylabel(ylbl)

    # Rescaled data + smoothing average
    ax = axes[1, 0]
    ax.plot(exp_times0, ylog, **style_raw, label='Log')
    ax.plot(exp_times0, ysmooth, **style_smooth, label='Smoothed')
    ax.set_ylabel(ylbl)
    ax.set_xlabel('Time (h)')
    # Same axes for all three plots after rescaling
    ax.set_ylim(ylims)
    ax.set_title('C', loc='left')
    ax.legend()

    fig.tight_layout()
    return fig, axes


def plot_dataset(df: dataset.CytokineDataset, cytokine: str):
    t1 = df.X.iloc[df.X.index.get_level_values('Cytokine') == cytokine]
    antigen = np.flatnonzero(np.array(t1.index.names) == 'Peptide')
    concentration = np.flatnonzero(np.array(t1.index.names) == 'Concentration')
    assert antigen.size != 0 and concentration.size != 0
    for index in t1.index:
        plt.plot(
            t1.loc[index],
            linestyle=CONCENTRATION_LINE[t1.loc[index].name[concentration[0]]],
            color=ANTIGEN_COLOUR[t1.loc[index].name[antigen[0]]]
        )

    plt.tight_layout()
    plt.show()


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
    plt.errorbar(X, avg, [df.mean(0) - df.min(0), df.max(0) - avg], fmt='.k', ecolor='gray',
                 lw=1, capsize=2)
    plt.errorbar(X, avg, df.std(0), fmt='ok', lw=3, capsize=4)

    plt.title('Learned Weights on Dataset 1')
    plt.xlabel('Weight')
    plt.ylabel('Value')

    # plt.show()
    plt.savefig(f'figure/weights_all_error.{file_format}')


def plot_loss(losses, title):
    plt.figure(figsize=(9.6, 4.8))

    for lr, loss in losses.items():
        train_loss, val_loss = loss
        plt.plot(train_loss, label='train: ' + str(lr))
        plt.plot(val_loss, label='val.: ' + str(lr))

    plt.title(f'Dataset(s) {title[0]} - Fold {title[1]}')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def plot_pred_concentration(preds: np.array, ds: dataset.CytokineDataset) -> None:
    # Create a dictionary to store values for each category
    cytokine_pred = {antigen: [] for antigen in ANTIGENS}

    df_plots = []
    for (x, _, r), pred in zip(list(ds[i] for i in range(len(ds))), preds):
        r = r[:, -1]
        antigen = np.argwhere(x.numpy() != 0.)[0][0]
        concentration = x.numpy()[antigen]
        cytokine_pred[ANTIGENS[antigen]].append([concentration, r.numpy(), pred])

        # We only consider the first cytokine output here (IFNg) for plotting.
        df_plots.append([ANTIGENS[antigen], concentration, pred[0], r.numpy()[0]])

    df_plots = pd.DataFrame(
        df_plots,
        columns=['Antigen Name', 'Input Concentration', 'Output Concentration', 'Predicted Output']
    )

    grouped_df = df_plots.groupby(['Antigen Name', 'Input Concentration']).agg({
        'Output Concentration': ['mean', 'max', 'min'],
        'Predicted Output': 'mean'
    }).reset_index()
    grouped_df.columns = ['Antigen Name', 'Input Concentration', 'Mean Output Concentration',
                          'Max Output Concentration', 'Min Output Concentration',
                          'Mean Predicted Concentration']
    grouped_df = grouped_df[~grouped_df['Antigen Name'].isin(['E1', 'G4'])]

    try:
        fig, axes = plt.subplots(nrows=1, ncols=len(grouped_df['Antigen Name'].unique()),
                                 figsize=(15, 3))

        for (antigen, group), ax in zip(grouped_df.groupby('Antigen Name'), axes):
            color = ANTIGEN_COLOUR.get(antigen, 'black')

            x = group['Input Concentration']
            y = group['Mean Output Concentration']

            lower_err = y - group['Min Output Concentration']
            upper_err = group['Max Output Concentration'] - y
            y_err = [lower_err, upper_err]

            ax.errorbar(x, y, yerr=y_err, fmt='o', label=antigen, color=color, capsize=5, alpha=0.2)
            ax.scatter(group['Input Concentration'], group['Mean Predicted Concentration'], color=color)

            ax.set_xlabel('Concentration')
            ax.set_xticks([-11., -10., -9., -8., -7., -6.])
            ax.set_ylabel('[Cytokine 1] Raw & Predicted Avg')
            ax.legend()

        plt.tight_layout()
        plt.show()

    except ValueError:
        pass
