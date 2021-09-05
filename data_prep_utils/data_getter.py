import pandas as pd
import functools


csv_loader = functools.partial(pd.read_csv, index_col=0)


def get_data_dir():
    from config import ROOT
    return ROOT / 'data'


def get_data():
    data_dir = get_data_dir()
    clinical_variables = csv_loader(data_dir / 'Clinical_Variables.csv')
    generic_alterations = csv_loader(data_dir / 'Genetic_alterations.csv')
    survival_time_event = csv_loader(data_dir / 'Survival_time_event.csv')
    treatment = csv_loader(data_dir / 'Treatment.csv')

    return clinical_variables, generic_alterations, survival_time_event, treatment
