from data_loader import get_data
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

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
    divisor = 3
    group_label = [0 for i in range(n_data)]
    for k in range(divisor):
        start = int(n_data / divisor * k)
        end = int(n_data/divisor*(k + 1))
        for idx in range(start, end):
            had_treatment = treatment.loc[idx, "Treatment"]
            group_label[sorted_survival_time_event.iloc[idx, 0]] = k + 1 if had_treatment else -(k + 1)

    plt.clf()
    plt.hist(group_label, range=(-divisor, divisor + 1), bins=2 * divisor + 1)
    plt.savefig("output2.png")

    generic_alterations = generic_alterations.iloc[:, 1:]
    clinical_variables = clinical_variables.iloc[:, 1:]
    generic_alterations = generic_alterations.join(clinical_variables)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(generic_alterations)
    principal_df = pd.DataFrame(data=principal_components, columns=['pc1', 'pc2'])
    principal_df["group"] = group_label

    print(principal_df)

    # principal axis
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(pd.DataFrame(data=pca.components_).transpose())

    plt.clf()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    groups = [i for i in range(-divisor, divisor + 1) if i != 0]
    for group in groups:
        indices_to_keep = principal_df["group"] == group
        ax.scatter(principal_df.loc[indices_to_keep, 'pc1'], principal_df.loc[indices_to_keep, 'pc2'], s=20)
    ax.legend(groups)
    plt.savefig("output3.png")
