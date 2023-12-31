import os, re
import glob
from typing import Union, Optional

import numpy as np
import pandas as pd
import scipy

import torch
from torch.utils import data

from scripts.utils import set_standard_order


def read_hdf_files(debug=True):
    data_path = (data_path := os.getcwd()) + '/data/processed/'

    naive_pairs = {
        'ActivationType': 'Naive',
        'Antibody': 'None',
        'APC': 'B6',
        'APCType': 'Splenocyte',
        'CARConstruct': 'None',
        'CAR_Antigen': 'None',
        'Genotype': 'WT',
        'IFNgPulseConcentration': 'None',
        'TCellType': 'OT1',
        'TLR_Agonist': 'None',
        'TumorCellNumber': '0k',
        'DrugAdditionTime': 36,
        'Drug': 'Null'
    }

    valid_names = ['TCellNumber', 'PeptideComparison', 'Activation', 'PeptideTumor']
    for file in os.listdir(data_path):
        valid = False
        for name in valid_names:
            if name in file:
                valid = True
                break

        if valid:
            if debug:
                print(file)

            if not file.lower().endswith('.hdf'):
                continue

            df: pd.DataFrame = pd.read_hdf(data_path + file)
            mask = [True] * df.shape[0]

            for index_name in df.index.names:
                if index_name in naive_pairs.keys():
                    mask = np.array(mask) & np.array([index == naive_pairs[index_name] for index in
                                                      df.index.get_level_values(index_name)])
                    df = df.droplevel([index_name])

            # add experiment name as multi-index level
            df = pd.concat([df[mask]], keys=[file[:-4]], names=['Data'])
            if 'df_full' not in locals():
                df_full = df.copy()
            else:
                df_full = pd.concat((df_full, df))

    return df_full


def load_dataset(level_values):
    df = read_hdf_files()
    df = df.stack().stack().to_frame('value')
    df = set_standard_order(df.reset_index())
    df = pd.DataFrame(df['value'].values,
                           index=pd.MultiIndex.from_frame(df.iloc[:, :-1]),
                           columns=['value'])
    df = df.loc[tuple(level_values), :].unstack(['Feature', 'Cytokine']).loc[:, 'value']
    # normalize the values in the time series
    df = (df - df.min()) / (df.max() - df.min())

    peptides = {k: v for v, k in enumerate(['N4', 'Q4', 'T4', 'V4', 'G4', 'E1'][::-1])}
    pep_dict = {}
    for peptide in peptides:
        if peptide in pd.unique(df.index.get_level_values('Peptide')):
            pep_dict[peptide] = peptides[peptide]

    # extract times and set classes
    cytokines = df.index.get_level_values('Peptide').map(pep_dict)

    return cytokines, df


def read_pickel_files(files: Optional[list[str]] = None):
    if files is None:
        files = glob.glob(os.path.join('data/final', '*.pkl'))

    return [pd.read_pickle(f) for f in files]


class CytokineDataset(data.Dataset):
    def __init__(self, folder: list[str], cytokine: list[str] = 'all', antigens: dict = None,
                 normalize: str = None, res_tolerance: float = 0.5):
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
                # 'A2': 6,
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
        for name in folder:
            self.pkl_files.extend([f for f in glob.glob(os.path.join('data/final', f'*{name}*.pkl'))])

        assert normalize is None or normalize == 'min-max'
        self.normalize = normalize

        self.dfs = {n: pd.read_pickle(f) for n, f in zip(folder, self.pkl_files)}
        self.logs = dict()
        self.smoothed = dict()
        self.splines = dict()
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
            lod = pd.read_json([f for f in glob.glob(os.path.join('data/lod', f'*{k}*.json'))][0])
            self.logs[k] = np.log10(self.dfs[k]).copy()
            for cytokine in self.classes.keys():
                self.logs[k].loc[self.logs[k].index.get_level_values("Cytokine") == cytokine] -= np.log10(lod.loc[cytokine].iloc[2])
                if normalize == 'min-max':
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
        self.X = self.X.sort_index(level=['Dataset', 'TCellNumber', 'Peptide', 'Concentration', 'Cytokine'])
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

        concentration = self.convert_unit(concentration_name)

        antigen = self.antigens[antigen_name]
        antigen_vec = self.ident_antigen[antigen] * concentration

        cytokine_measure = self.X.iloc[idx, :].to_numpy()
        cytokine_measure = torch.from_numpy(cytokine_measure)

        r = self.X.iloc[
            (self.X.index.get_level_values('Peptide') == antigen_name) &
            (self.X.index.get_level_values('Concentration') == concentration_name) &
            (self.X.index.get_level_values('Dataset') == dataset_name) &
            (self.X.index.get_level_values('TCellNumber') == tcell_count)
        ]
        r = r.droplevel('Dataset').droplevel('TCellNumber')

        # Returns just the values at time 64.0 because those are the only ones we care about.
        return antigen_vec.float(), cytokine_measure.float(), torch.tensor(r.get(64.).values).float()

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


def get_train_test_subset(params, train_per: float = 0.7, val_per: float = 0.15) -> list[data.Subset]:
    assert np.less(train_per + val_per, 1.)

    df = [f'PeptideComparison_{i}' for i in range(1, 10)] if params['df'] == 'all' \
        else [f'PeptideComparison_{params["df"]}']
    df = CytokineDataset(df)

    train_num = int(train_per * len(df))
    valid_num = int(val_per * len(df))
    test_num = len(df) - (train_num + valid_num)
    return data.random_split(df, [train_num, valid_num, test_num])
