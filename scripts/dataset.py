import os, re
import glob

import numpy as np
import pandas as pd
import scipy

import torch
from sklearn.model_selection import KFold
from torch.utils import data


class CytokineDataset(data.Dataset):
    def __init__(self, folder: str = 'data/final', files: list[str] = None, cytokine: list[str] = 'all',
                 antigens: dict = None, norm: str = None, res_tolerance: float = 0.5):
        if cytokine != 'all':
            self.classes = {cyto: i for i, cyto in enumerate(cytokine)}
        else:
            self.classes = {
                'IFNg': 0,
                'IL-17A': 1,
                'IL-2': 2,
                'IL-6': 3,
                'TNFa': 4,
            }

        self.ident = torch.eye(len(self.classes))

        if antigens is None:
            self.antigens = {
                'N4': 0,
                'Q4': 1,
                'T4': 2,
                'V4': 3,
                'G4': 4,
                'E1': 5,
            }
        else:
            self.antigens = antigens
        self.ident_antigen = torch.eye(len(self.antigens))

        self.concentrations = {
            '1uM': 1e-6,
            '100pM': 1e-7,
            '10pM': 1e-8,
            '1pM': 1e-9,
            '100nM': 1e-10,
            '10nM': 1e-11,
            '1nM': 1e-12,
        }

        self.pkl_files = []
        if files is not None:
            for name in files:
                self.pkl_files.extend([f for f in glob.glob(os.path.join(folder, f'*{name}*.pkl'))])
        else:
            self.pkl_files = glob.glob(os.path.join(folder, '*'))

        assert norm is None or norm == 'min-max'
        self.norm = norm
        self.x_min, self.x_max = -np.inf, np.max

        self.dfs = {n: pd.read_pickle(f) for n, f in zip(folder, self.pkl_files)}
        self.logs = dict()
        self.smoothed = dict()
        self.splines = dict()

        levels = set()
        for k in self.dfs.keys():
            levels = levels.union(self.dfs[k].index.names)

        for k in self.dfs.keys():
            if len(new_levels := levels.difference(self.dfs[k].index.names)) != 0:
                idx = self.dfs[k].index.to_frame()
                for level in new_levels:
                    idx.insert(0, level, '')
                self.dfs[k].index = pd.MultiIndex.from_frame(idx)

        for k in self.dfs.keys():
            # this seems unnecessary but is done in the paper for some reason
            self.dfs[k] = self.dfs[k].stack().unstack('Cytokine')
            df_zero = (np.sum(self.dfs[k] == self.dfs[k].min(), axis=1) == len(self.dfs[k].columns)).unstack('Time')
            for t in range(len(df_zero.columns) - 1, -1, -1):
                for row in range(len(df_zero)):
                    if (not df_zero.iloc[row, 0: t].all()) and df_zero.iloc[row, t]:
                        self.dfs[k].loc[tuple(list(df_zero.iloc[row, :].name) + [df_zero.columns[t]])] = np.nan
            self.dfs[k] = self.dfs[k].interpolate(method='linear').stack('Cytokine').unstack('Time')

            # to be used later in the normalization process
            df_max = self.dfs[k].values.max()

            # get valid antigens and cytokines
            self.dfs[k].query(
                ' | '.join(f'(@self.dfs[@k].index.get_level_values("Peptide") == "{antigen}")' for antigen in self.antigens),
                engine='python', inplace=True
            )
            self.dfs[k].query(
                ' | '.join(f'(@self.dfs[@k].index.get_level_values("Cytokine") == "{cyto}")' for cyto in self.classes),
                engine='python', inplace=True
            )

            # normalize and log the dataset
            if len(lod := [f for f in glob.glob(os.path.join('data/lod', f'*{k}*.json'))]) != 0:
                lod = pd.read_json(lod[0])
            else:
                lod = None

            self.logs[k] = np.log10(self.dfs[k]).copy()
            for cytokine in self.classes.keys():
                idx = self.logs[k].index.get_level_values("Cytokine") == cytokine
                if lod is not None:
                    self.logs[k].loc[idx] -= np.log10(lod.loc[cytokine].iloc[2])
                else:
                    self.logs[k].loc[idx] -= np.log10(self.dfs[k].loc[idx].values.min())

                if norm == 'min-max':
                    self.logs[k].loc[self.logs[k].index.get_level_values("Cytokine") == cytokine] /= (
                            np.log10(df_max) - np.log10(lod.loc[cytokine].iloc[2])
                    )

            self.smoothed[k] = self.logs[k].copy()
            self.smoothed[k].iloc[:, 1: -1] = self.smoothed[k].rolling(window=3, center=True, axis=1).mean().iloc[:, 1: -1]

            self.logs[k].insert(0, 0., 0.)
            self.smoothed[k].insert(0, 0., 0.)

            self.splines[k] = pd.DataFrame(None, index=self.dfs[k].unstack('Cytokine').index, columns=list(self.classes.keys()), dtype=object)
            for cytokine in self.splines[k].columns:
                for row in self.splines[k].index:
                    y = self.smoothed[k].loc[tuple(list(row) + [cytokine])]
                    r = self.logs[k].loc[tuple(list(row) + [cytokine])]
                    self.splines[k].loc[row, cytokine] = scipy.interpolate.UnivariateSpline(self.smoothed[k].columns, y, s=res_tolerance * np.sum((y - r) ** 2))

        self.X = pd.concat(self.dfs, names=['Dataset'])
        self.X = self.X.sort_index(axis=1)
        aux_levels = list(levels.difference({'Dataset', 'Peptide', 'Concentration', 'Cytokine'}))
        self.X = self.X.sort_index(level=['Dataset',  *aux_levels, 'Peptide', 'Concentration', 'Cytokine'])
        for row in self.X.index:
            spline = self.splines[row[0]].loc[row[1: -1]][row[-1]]
            self.X.loc[row] = np.array([spline.integral(0, t) for t in self.X.columns])

        # ensures the integral is monotonic and non-negative
        for t in range(len(self.X.columns)):
            self.X.iloc[:, t] -= np.nansum(self.X.diff(axis=1)[self.X.diff(axis=1) < 0].loc[:, self.X.columns[: t+1]], axis=1)
        self.X[self.X <= 0.] = 0.

    def __len__(self):
        assert self.X.shape[0] / len(self.classes) == self.X.shape[0] // len(self.classes)
        return self.X.shape[0] // len(self.classes)

    def __getitem__(self, idx: int, min_max: tuple[float, float] = None):
        idx = idx * len(self.classes)

        dataset_name, tcell_count, antigen_name, concentration_name, _ = self.X.iloc[idx, :].name

        antigen = self.antigens[antigen_name]
        antigen_vec = self.ident_antigen[antigen] * self.convert_unit(concentration_name)

        cytokine_measure = self.X.iloc[idx, :].to_numpy()
        cytokine_measure = torch.from_numpy(cytokine_measure)

        r = self.X.iloc[
            (self.X.index.get_level_values('Peptide') == antigen_name) &
            (self.X.index.get_level_values('Concentration') == concentration_name) &
            (self.X.index.get_level_values('Dataset') == dataset_name) &
            (self.X.index.get_level_values('TCellNumber') == tcell_count)
        ]
        r = r.droplevel('Dataset').droplevel('TCellNumber')

        return antigen_vec.float(), cytokine_measure.float(), torch.tensor(r.iloc[:, : 30].to_numpy()).float()

    @staticmethod
    def convert_unit(c: str) -> float:
        match = re.match(r'([0-9]+)([a-z]+)', c, re.I)
        if match:
            items = match.groups()
        else:
            raise ValueError

        amount = float(items[0])
        unit = items[1][0]
        if unit == 'n':
            amount = amount * 1e-9
        elif unit == 'p':
            amount = amount * 1e-12
        elif unit == 'u':
            amount = amount * 1e-6

        return np.log10(amount)
