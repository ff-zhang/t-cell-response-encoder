import os
import re
import glob
from typing import List, Union, Optional

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

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


def load_dataset(level_values, return_df=True, debug=True):
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


def read_pickel_files(files: Optional[List[str]] = None):
    if files is None:
        files = glob.glob(os.path.join('data/final', '*.pkl'))

    return [pd.read_pickle(f) for f in files]


class CytokineDataset(Dataset):
    def __init__(self, folder: Union[str, List[str]], cytokine: str = 'all', antigens: dict = None):
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

        if type(folder) is str and os.path.isdir(folder):
            self.pkl_files = glob.glob(os.path.join(folder, '*.pkl'))
        elif type(folder) is list:
            self.pkl_files = []
            for name in folder:
                self.pkl_files.extend([f for f in glob.glob(os.path.join('data/final', f'*{name}*.pkl'))])
        else:
            raise TypeError

        # TODO: handle case when label and file isn't 1-1
        self.dfs = {n: pd.read_pickle(f) for n, f in zip(folder, self.pkl_files)}

        self.X = pd.concat(self.dfs, names=['Dataset'])

        if cytokine != 'all':
            self.X = self.X.iloc[(self.X.index.get_level_values('Cytokine') == cytokine)]

        # get valid antigens and cytokines
        self.X = self.X.query(
            ' | '.join(f'(@self.X.index.get_level_values("Peptide") == "{antigen}")' for antigen in self.antigens)
        )
        self.X = self.X.query(
            ' | '.join(f'(@self.X.index.get_level_values("Cytokine") == "{cyto}")' for cyto in self.classes)
        )

        self.X = self.X.sort_index(axis=1)
        self.X = self.X.sort_index(level=['Dataset', 'TCellNumber', 'Peptide', 'Concentration', 'Cytokine'])
        self.X = self.X.cumsum(axis=1).interpolate(axis=1)

    def __len__(self):
        assert self.X.shape[0] / 5 == self.X.shape[0] // 5
        return self.X.shape[0] // 5

    def __getitem__(self, idx):
        idx = idx * 5

        dataset_name, _, tcell_count, antigen_name, concentration_name = self.X.iloc[idx, :].name

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
        r = r.droplevel('Dataset').droplevel('TCellNumber').to_numpy()

        return antigen_vec.float(), cytokine_measure.float(), torch.from_numpy(r).float()

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
