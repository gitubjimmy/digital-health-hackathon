from data_loader import get_data

if __name__ == '__main__':
    clinical_variables, generic_alterations, survival_time_event, treatment = get_data()
    n_data = treatment.shape[0]
    for idx in range(n_data):
        if survival_time_event.loc[idx, "event"] == 0:
            survival_time_event.at[idx, "time"] = 999
    hist = survival_time_event.hist(column="time", range=(0, 200), bins=10)
    fig = hist[0][0].get_figure()
    fig.savefig("output.png")
