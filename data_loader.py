import pandas as pd

def get_data():
    clinical_variables = pd.read_csv('./data/Clinical_Variables.csv')
    generic_alterations = pd.read_csv('./data/Genetic_alterations.csv')
    survival_time_event = pd.read_csv('./data/Survival_time_event.csv')
    treatment = pd.read_csv('./data/Treatment.csv')

    return clinical_variables, generic_alterations, survival_time_event, treatment
