import os, re
import glob

import numpy as np
import pandas as pd
import scipy

import torch
from torch.utils import data

TIME_SERIES_LENGTH = 64

DEFAULT_ANTIGENS = ['N4', 'Q4', 'T4', 'V4', 'G4', 'E1']
DEFAULT_CYTOKINES = ['IFNg', 'IL-17A', 'IL-2', 'IL-6', 'TNFa']
DEFAULT_CONCENTRATIONS = ['100pM', '10pM', '300nM', '100nM', '30nM', '10nM', '3nM', '1nM', '1uM']


class CytokineDataset(data.Dataset):
    def __init__(self, folder: str = 'data/final', files: list[str] = None, cytokines: list = None,
                 antigens: list = None, concentrations: list = None, exclude: dict = None,
                 lod: bool = False, norm: bool = True, rtol: float = 0.5):
        """
        :param folder: root directory to search for dataset .pkl files
        :param files: list of (substrings of) files names to load
        :param cytokine: cytokines labels to include in the loaded dataset
        :param antigens: antigen labels to include in the loaded dataset
        :param exclude: labels to be excluded from the loaded dataset
        :param lod: determines whether to normalize by the LoD or dataset minimum
        :param norm: determines whether to normalize the dataset into the range [-1, 1] at the end
        :param rtol: the fraction of the sum of squared residuals between raw and smoothed data used as a tolerance during spline fitting
        """

        self._init_params(folder, files, cytokines, antigens, concentrations, exclude)
        self._process_datasets(lod, rtol)

        # Combines the individual dataframes into one.
        self.X = pd.concat(self.dfs, names=['Dataset'])
        self.X = self.X.sort_index(axis=1)
        aux_levels = list(self.levels.difference({'Dataset', 'Peptide', 'Concentration', 'Cytokine'}))
        self.X = self.X.sort_index(level=['Dataset', *aux_levels, 'Peptide', 'Concentration', 'Cytokine'])
        for row in self.X.index:
            spline = self.splines[row[0]].loc[row[1: -1]][row[-1]]
            self.X.loc[row] = np.array([spline.integral(0, t) for t in self.X.columns])

        # Ensures the integral is monotonic and non-negative
        for t in range(len(self.X.columns)):
            self.X.iloc[:, t] -= np.nansum(self.X.diff(axis=1)[self.X.diff(axis=1) < 0].loc[:, self.X.columns[: t+1]], axis=1)
        self.X[self.X <= 0.] = 0.

        # Normalize the dataset to be in the interval [-1, 1].
        if norm == 'min-max':
            self.x_min, self.x_max = self.X.min(), self.X.max()
            self.X = (self.X - self.x_min) / (self.x_max - self.x_min) * 2 - 1

    def _init_params(self, folder: str = 'data/final', files: list[str] = None, cytokines: list = None,
                     antigens: dict = None, concentrations: list = None, exclude: dict = None):
        # Checks that either the elements of the labels and targets are specified, or the elements
        # to be excluded are specified, not both.
        assert cytokines is None or exclude is None
        assert antigens is None or exclude is None

        self.classes = cytokines if cytokines is not None else DEFAULT_CYTOKINES
        self.antigens = antigens if antigens is not None else DEFAULT_ANTIGENS
        self.concs = concentrations if concentrations is not None else DEFAULT_CONCENTRATIONS

        # Removes unwanted values from their respective class attributes.
        # These will be used to filter out data points with the unwanted values latter.
        if exclude is not None:
            attr_dict = {'Cytokine': self.classes, 'Peptide': self.antigens, 'Concentration': self.concs}
            assert exclude.keys() <= attr_dict.keys()
            for k, v in exclude.items():
                if type(v) is list or type(v) is tuple:
                    for w in v:
                        attr_dict[k].remove(w)
                elif type(v) is str:
                    attr_dict[k].remove(v)
                else:
                    raise TypeError

        self.classes = {k: v for v, k in enumerate(self.classes)}
        self.antigens = {k: v for v, k in enumerate(self.antigens)}

        self.ident = torch.eye(len(self.classes))
        self.ident_antigen = torch.eye(len(self.antigens))

        self.pkl_files = []
        if files is not None:
            for name in files:
                self.pkl_files.extend([f for f in glob.glob(os.path.join(folder, f'*{name}*.pkl'))])
        else:
            self.pkl_files = glob.glob(os.path.join(folder, '*'))
        self.dfs = {n: pd.read_pickle(f) for n, f in enumerate(self.pkl_files)}

        # Sets up data structures to hold intermediate data sets in the processing pipeline.
        self.logs = dict()
        self.smoothed = dict()
        self.splines = dict()

        # Gets all index levels (i.e. columns) across the different data frames.
        self.levels = set()
        for k in self.dfs.keys():
            self.levels = self.levels.union(self.dfs[k].index.names)

    def _process_datasets(self, lod, rtol):
        # Inserts placeholder index levels to ensure easy data frame concatenation later.
        for k in self.dfs.keys():
            if len(new_levels := self.levels.difference(self.dfs[k].index.names)) != 0:
                idx = self.dfs[k].index.to_frame()
                for level in new_levels:
                    idx.insert(0, level, '')
                self.dfs[k].index = pd.MultiIndex.from_frame(idx)

        for k in self.dfs.keys():
            print(f'Processing file: {self.pkl_files[k]}')

            # This step seems unnecessary but is done in the paper the data is taken from.
            self.dfs[k] = self.dfs[k].stack().unstack('Cytokine')
            df_zero = (np.sum(self.dfs[k] == self.dfs[k].min(), axis=1) == len(
                self.dfs[k].columns)).unstack('Time')
            for t in range(len(df_zero.columns) - 1, -1, -1):
                for row in range(len(df_zero)):
                    if (not df_zero.iloc[row, 0: t].all()) and df_zero.iloc[row, t]:
                        self.dfs[k].loc[
                            tuple(list(df_zero.iloc[row, :].name) + [df_zero.columns[t]])] = np.nan
            self.dfs[k] = self.dfs[k].interpolate(method='linear').stack('Cytokine').unstack('Time')

            # Gets data points with valid antigens, cytokines, and concentrations using our
            # initialized class attributes.
            self.dfs[k].query(
                ' | '.join(
                    f'(@self.dfs[@k].index.get_level_values("Peptide") == "{antigen}")' for antigen
                    in self.antigens),
                engine='python', inplace=True
            )
            self.dfs[k].query(
                ' | '.join(
                    f'(@self.dfs[@k].index.get_level_values("Cytokine") == "{cyto}")' for cyto in
                    self.classes),
                engine='python', inplace=True
            )
            self.dfs[k].query(
                ' | '.join(
                    f'(@self.dfs[@k].index.get_level_values("Concentration") == "{conc}")' for conc
                    in self.concs),
                engine='python', inplace=True
            )

            # Skips the rest of the processing if there are no valid points in the data frame.
            if len(self.dfs[k]) == 0:
                continue

            # Normalizes the lower limit of the points.
            if lod and len(
                    lod := [f for f in glob.glob(os.path.join('data/lod', f'*{k}*.json'))]) != 0:
                lod = pd.read_json(lod[0])
            else:
                lod = None

            # Take the log of the current data frame.
            self.logs[k] = np.log10(self.dfs[k]).copy()
            for cytokine in self.classes.keys():
                idx = self.logs[k].index.get_level_values("Cytokine") == cytokine
                if lod is not None:
                    self.logs[k].loc[idx] -= np.log10(lod.loc[cytokine].iloc[2])
                else:
                    self.logs[k].loc[idx] -= np.log10(self.dfs[k].loc[idx].values.min())

            # Take the weighted rolling average of points to smooth the curves.
            self.smoothed[k] = self.logs[k].copy()
            self.smoothed[k].iloc[:, 1: -1] = self.smoothed[k].rolling(window=3, center=True,
                                                                       axis=1).mean().iloc[:, 1: -1]

            # Reinserts the initial time step.
            self.logs[k].insert(0, 0., 0.)
            self.smoothed[k].insert(0, 0., 0.)

            # Computes a spline for each data point in the data frame of smoothed curves.
            self.splines[k] = pd.DataFrame(None, index=self.dfs[k].unstack('Cytokine').index,
                                           columns=list(self.classes.keys()), dtype=object)
            for cytokine in self.splines[k].columns:
                for row in self.splines[k].index:
                    y = self.smoothed[k].loc[tuple(list(row) + [cytokine])]
                    r = self.logs[k].loc[tuple(list(row) + [cytokine])]

                    self.splines[k].loc[row, cytokine] = scipy.interpolate.UnivariateSpline(
                        self.smoothed[k].columns, y, s=rtol * np.sum((y - r) ** 2)
                    )

    def __len__(self):
        assert self.X.shape[0] / len(self.classes) == self.X.shape[0] // len(self.classes)
        return self.X.shape[0] // len(self.classes)

    def __getitem__(self, idx: int, min_max: tuple[float, float] = None):
        idx = idx * len(self.classes)

        dataset_name, act_type, tcell_count, antigen_name, conc_name, _ = self.X.iloc[idx, :].name

        antigen = self.antigens[antigen_name]
        normalized_conc = (self.convert_unit(conc_name) + 12) / (4 - 12)
        antigen_vec = self.ident_antigen[antigen] * normalized_conc

        cytokine_measure = self.X.iloc[idx, :].to_numpy()
        cytokine_measure = torch.from_numpy(cytokine_measure)

        r = self.X.iloc[
            (self.X.index.get_level_values('Dataset') == dataset_name) &
            (self.X.index.get_level_values('ActivationType') == act_type) &
            (self.X.index.get_level_values('TCellNumber') == tcell_count) &
            (self.X.index.get_level_values('Peptide') == antigen_name) &
            (self.X.index.get_level_values('Concentration') == conc_name)
        ]
        r = r.droplevel('Dataset').droplevel('TCellNumber')

        return antigen_vec.float(), cytokine_measure.float(), torch.tensor(r.iloc[:, : TIME_SERIES_LENGTH].to_numpy()).float()

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
