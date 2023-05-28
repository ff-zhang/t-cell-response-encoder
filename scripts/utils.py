#! /usr/bin/env python3
""" Adapt dataframes to processing pipeline

- Update filenames, level names and level values
- Retrieve standard ordering for T cell number, peptide and concentration

Filename changes
        TCellTypeComparison_OT1,P14,F5_Timeseries_3
                No F5 in this dataset, remove mention of F5 in filename

        NaiveVsExpandedTCells_OT1_Timeseries_1
                Change filename to Activation_Timeseries_1

        CD4_5CC7_2
                Change filename to TCellType_OT1_CD4_5CC7_Timeseries_2

        AntagonismComparison_OT1_Timeseries_1
                Change filename to OT1_Antagonism_1

Level name changes
        APC_Type, APC Type --> APCType
        Agonist, TCR_Antigen --> Peptide

Level value changes
        Activation_TCellNumber_1
                TCellType --> ActivationType

        Tumor datasets
                Include IFNg pulse concentration with Nones for Splenocytes
                Adapt concentrations
                Change TCell/TumorCellNumber format
                Add APC & APCType level TODO: APC is B16?

        CAR datasets
                Genotype: Naive --> WT
                Concentration: 1uM

        ITAMDef datasets
                Include TCellNumber (30k, not standard 100k)

        TCellType datasets
                Peptide2: NotApplicable --> None

        Antagonism datasets
                Select nonantagonist data
"""

import os
import re
import warnings

import pandas as pd

warnings.filterwarnings('ignore')

path = (path := os.getcwd()) + '/data/current/'

filename_changes = {
    'TCellTypeComparison_OT1,P14,F5_Timeseries_3': 'TCellTypeComparison_OT1,P14_Timeseries_3',
    'NaiveVsExpandedTCells_OT1_Timeseries_1': 'Activation_Timeseries_1',
    'CD4_5CC7_2': 'TCellType_OT1_5CC7_Timeseries_2',
    'PeptideComparison_OT1_Timeseries_20': 'NewPeptideComparison_OT1_Timeseries_20'
}

level_name_changes = {
    'APC Type': 'APCType',
    'APC_Type': 'APCType',
    'Agonist': 'Peptide',
    'TCR_Antigen': 'Peptide'
}

tumor_timeseries = [
    'TumorTimeseries_1',
    'TumorTimeseries_2',
    'PeptideTumorComparison_OT1_Timeseries_1',
    'PeptideTumorComparison_OT1_Timeseries_2'
]

activation_timeseries = ['Activation_TCellNumber_1']

itamdef_timeseries = ['ITAMDeficient_OT1_Timeseries_%d' % num for num in [9, 10, 11]]

tcelltype_timeseries = ['TCellTypeComparison_OT1,P14,F5_Timeseries_3']


def sort_SI_column(columnValues, unitSuffix):
    si_prefix_dict = {'a': 1e-18, 'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3, '': 1e0}
    numericValues = []
    for val in columnValues:
        val = val.replace(' ', '')
        splitString = re.split('(\d+)', val)[1:]
        # If no numeric variables, assume zero
        if len(splitString) < 2:
            numeric_val = 0
        else:
            # If no numeric variables, assume zero
            if splitString[1] == '':
                numeric_val = 0
            else:
                # If correctly formatted in SI, use SI prefix dict
                if len(splitString[1]) < 3:
                    siUnit = splitString[1].split(unitSuffix)[0]
                # If unit are strange assign lowest SI prefix to shunt to the end of the order
                else:
                    siUnit = 'a'
                numeric_val = si_prefix_dict[siUnit] * float(splitString[0])

        numericValues.append(numeric_val)

    return numericValues


def return_data_date_dict():
    dataFolder = (dataFolder := os.getcwd()) + '/data/final'
    dateDict = {}
    for fileName in os.listdir(dataFolder):
        if '.pkl' in fileName:
            datasetDate = fileName.split('-')[1]
            datasetName = fileName.split('-')[2]
            dateDict[datasetName] = float(datasetDate) * -1

    return dateDict


def set_standard_order(df, returnSortedLevelValues=False):
    levelDict = {'DATASORT': 'Data', 'TCELLSORT': 'TCellNumber', 'PEPTIDESORT': 'Peptide',
                 'CONCSORT': 'Concentration'}
    peptide_dict = {'N4': 13, 'A2': 12, 'Y3': 11, 'Q4': 10, 'T4': 9, 'V4': 8, 'G4': 7, 'E1': 6,
                    'AV': 5, 'A3V': 4, 'gp33WT': 3, 'None': 2}
    levelsToSort = []
    if 'Data' in df.columns:
        dataset_dict = return_data_date_dict()
        df['DATASORT'] = df.Data.map(dataset_dict)
        levelsToSort.append('DATASORT')
    if 'TCellNumber' in df.columns:
        df['TCELLSORT'] = df.TCellNumber.str.replace('k', '').astype('float')
        levelsToSort.append('TCELLSORT')
    if 'Peptide' in df.columns:
        df['PEPTIDESORT'] = df.Peptide.map(peptide_dict)
        levelsToSort.append('PEPTIDESORT')
    if 'Concentration' in df.columns:
        df['CONCSORT'] = sort_SI_column(df.Concentration, 'M')
        levelsToSort.append('CONCSORT')
    sortColumnRemovalVar = -1 * len(levelsToSort)

    if not returnSortedLevelValues:
        df = df.sort_values(levelsToSort, ascending=False).iloc[:, :sortColumnRemovalVar]
        return df
    else:
        sortedLevelValues = {}
        for levelToSort in levelsToSort:
            levelValues = list(pd.unique(
                df.sort_values([levelToSort], ascending=False).loc[:, levelDict[levelToSort]]))
            sortedLevelValues[levelDict[levelToSort]] = levelValues

        return sortedLevelValues


def createLabelDict(df, discretized_time=False, sortedValues={}):
    # fulldf = df.stack()
    fulldf = df.copy()
    labelDict = {}
    for i in range(fulldf.index.nlevels):
        levelName = fulldf.index.levels[i].name
        parameterList = ['Event', 'event']
        if not discretized_time:
            parameterList += ['Time']
        if levelName not in parameterList:
            labelDict[levelName] = list(pd.unique(fulldf.index.get_level_values(levelName)))
    if len(sortedValues) != 0:
        for level in sortedValues:
            labelDict[level] = sortedValues[level]

    return labelDict
