from sklearn.decomposition import PCA

from data_loader import get_data
from utils import write_to_csv
import numpy as np
import pandas as pd


def find_rel_nodes(node_idx, group):
    vt[node_idx] = 1
    for other_node_idx in range(300):
        if not vt[other_node_idx] and W[node_idx][other_node_idx] > 0.9:
            group.append(other_node_idx)
            find_rel_nodes(other_node_idx, group)
    return group


if __name__ == '__main__':
    clinical_variables, generic_alterations, survival_time_event, treatment = get_data()
    A = [[0 for _ in range(300)] for _ in range(300)]
    beta = 12
    amp = 1
    for gene_idx in range(300):
        gene_name = f'G{gene_idx + 1}'
        gene_col = generic_alterations[gene_name]
        for comparing_gene_idx in range(300):
            comparing_gene_name = f'G{comparing_gene_idx + 1}'
            comparing_gene_col = generic_alterations[comparing_gene_name]
            corr = gene_col.corr(comparing_gene_col)
            A[gene_idx][comparing_gene_idx] = (0.5 + corr / 2) ** beta * amp

    L = np.dot(A, A)
    k = [sum(row) for row in A]
    W = [[0 for _ in range(300)] for _ in range(300)]
    for i in range(300):
        L[i][i] -= 1
    for i in range(300):
        for j in range(300):
            W[i][j] = (L[i][j] + A[i][j]) / (min(k[i], k[j]) + 1 - A[i][j])
            if i != j:
                W[i][j] *= 1000

    vt = [0 for _ in range(300)]
    groups = []
    for idx in range(300):
        if not vt[idx]:
            groups.append(find_rel_nodes(idx, [idx]))

    # Imputing biased survival time
    # for row_idx in range(survival_time_event.shape[0]):
    #     if survival_time_event.loc[row_idx, "event"] == 0:
    #         current_survival_time = survival_time_event.loc[row_idx, "time"]
    #         survival_time_event.at[row_idx, "time"] = survival_time_event.loc[
    #             (survival_time_event["event"] == 1) &
    #             (survival_time_event["time"] > current_survival_time), "time"].mean()

    for gp in groups:
        gp.sort()
        gene_column_names = [f'G{gene_idx + 1}' for gene_idx in gp]
        group_alteration_data = generic_alterations[gene_column_names]
        coefficient = []
        if len(gp) > 1:
            pca = PCA(n_components=2)
            pca.fit_transform(group_alteration_data)
            coefficient = pca.components_[0]
        else:
            coefficient = [1]
        print('Group', groups.index(gp))
        for idx in range(len(gene_column_names)):
            print(gene_column_names[idx], coefficient[idx])
