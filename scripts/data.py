import os

import numpy as np
import pandas
import pandas as pd

from scripts.utils import createLabelDict, set_standard_order


def read_hdf_files(debug=True):
    data = (data := os.getcwd()) + '/data/processed/'

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
    for file in os.listdir(data):
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

            df: pandas.DataFrame = pd.read_hdf(data + file)
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

    peptides = {k: v for v, k in enumerate(["N4", "Q4", "T4", "V4", "G4", "E1"][::-1])}
    pep_dict = {}
    for peptide in peptides:
        if peptide in pd.unique(df.index.get_level_values("Peptide")):
            pep_dict[peptide] = peptides[peptide]

    # extract times and set classes
    cytokines = df.index.get_level_values("Peptide").map(pep_dict)

    return cytokines, df
