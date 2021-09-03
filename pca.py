from data_loader import get_data
import matplotlib.pyplot as plt

if __name__ == '__main__':
    clinical_variables, generic_alterations, survival_time_event, treatment = get_data()
    n_data = treatment.shape[0]
    for idx in range(n_data):
        if survival_time_event.loc[idx, "event"] == 0:
            survival_time_event.at[idx, "time"] = 999
    hist = survival_time_event.hist(column="time", range=(0, 200), bins=10)
    fig = hist[0][0].get_figure()
    fig.savefig("output.png")

    sorted_survival_time_event = survival_time_event.sort_values(by="time")
    divisor = 5
    group_label = [0 for i in range(n_data)]
    for k in range(divisor):
        start = int(n_data / divisor * k)
        end = int(n_data/divisor*(k + 1))
        for idx in range(start, end):
            had_treatment = treatment.loc[idx, "Treatment"]
            group_label[sorted_survival_time_event.iloc[idx, 0]] = k + 1 if had_treatment else -(k + 1)

    plt.clf()
    plt.hist(group_label, range=(-5, 6), bins=11)
    plt.savefig("output2.png")

