import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dataset


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
    antigen_colour = {
        'N4': 'blue',
        'Q4': 'orange',
        'T4': 'green',
        'V4': 'red',
        'G4': 'purple',
        'E1': 'brown',
    }
    concentration_colour = {
        '100pM': '',
        '10pM': '',
        '100nM': ':',
        '10nM': '-.',
        '1nM': '--',
        '1uM': '-',
    }
    t1 = df.X.iloc[df.X.index.get_level_values('Cytokine') == cytokine]
    for index in t1.index:
        assert(t1.loc[index].name[4] == cytokine)
        curve = t1.loc[index]
        plt.plot(curve, linestyle=concentration_colour[t1.loc[index].name[3]],
                 color=antigen_colour[t1.loc[index].name[2]])

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


if __name__ == '__main__':
    df = dataset.CytokineDataset([f'PeptideComparison_{i}' for i in range(1, 10)])
    ds = 'PeptideComparison_1'
    plot_spline_process_steps(('100k', 'N4', '10nM', 'IL-2'), df.dfs[ds], df.logs[ds], df.smoothed[ds], df.splines[ds])
