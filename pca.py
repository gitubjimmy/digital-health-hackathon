from data_loader import get_data
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import networkx as nx
import numpy as np

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
        end = int(n_data / divisor * (k + 1))
        for idx in range(start, end):
            had_treatment = treatment.loc[idx, "Treatment"]
            group_label[sorted_survival_time_event.iloc[idx, 0]] = k + 1 if had_treatment else -(k + 1)

    plt.clf()
    plt.hist(group_label, range=(-divisor, divisor + 1), bins=2 * divisor + 1)
    plt.savefig("output2.png")

    # gene relativity graph
    relativity_graph = [[0 for _ in range(300)] for _ in range(300)]
    relativity_graph_xor = [[0 for _ in range(300)] for _ in range(300)]
    mutation_per_person = []
    for data_idx in range(n_data):
        mutated_genes = generic_alterations.loc[data_idx, generic_alterations.loc[data_idx] == 1].index
        mutated_gene_indices = [mutated_gene.split('G')[1] for mutated_gene in mutated_genes if 'G' in mutated_gene]
        mutated_gene_indices = [int(gene_idx) for gene_idx in mutated_gene_indices if gene_idx.isnumeric()]
        mutation_per_person.append(len(mutated_gene_indices))

        normal_genes = generic_alterations.loc[data_idx, generic_alterations.loc[data_idx] == 0].index
        normal_gene_indices = [normal_gene.split('G')[1] for normal_gene in normal_genes if 'G' in normal_gene]
        normal_gene_indices = [int(gene_idx) for gene_idx in normal_gene_indices if gene_idx.isnumeric()]
        for start_idx in range(len(mutated_gene_indices)):
            start_gene_idx = mutated_gene_indices[start_idx] - 1
            for end_idx in range(start_idx + 1, len(mutated_gene_indices)):
                end_gene_idx = mutated_gene_indices[end_idx] - 1
                relativity_graph[start_gene_idx][end_gene_idx] += 1
                relativity_graph[end_gene_idx][start_gene_idx] = relativity_graph[start_gene_idx][end_gene_idx]

            for end_idx in range(len(normal_gene_indices)):
                end_gene_idx = normal_gene_indices[end_idx] - 1
                relativity_graph_xor[start_gene_idx][end_gene_idx] += 1
    to_csv = '\n'.join([','.join([str(cell) for cell in row]) for row in relativity_graph])
    to_csv_xor = '\n'.join([','.join([str(cell) for cell in row]) for row in relativity_graph_xor])
    with open('./data/relativity_graph.csv', 'w') as output_csv:
        output_csv.write(to_csv)
    with open('./data/relativity_graph_xor.csv', 'w') as output_csv:
        output_csv.write(to_csv_xor)
    print('average_mutation_per_person', sum(mutation_per_person)/len(mutation_per_person))
    flat_relativity_graph_xor = [cell_value for row in relativity_graph_xor for cell_value in row if cell_value > 0]
    print('min value in relativity_graph_xor', min(flat_relativity_graph_xor))

    # effective genes
    gene_effectiveness = []
    for gen_idx in range(300):
        column_name = f"G{gen_idx + 1}"
        mutated_indices = generic_alterations.loc[generic_alterations[column_name] == 1, column_name].index
        normal_indices = generic_alterations.loc[generic_alterations[column_name] == 0, column_name].index
        survived_time_mutated = survival_time_event.loc[mutated_indices, "time"].mean()
        survived_time_normal = survival_time_event.loc[normal_indices, "time"].mean()
        gene_effectiveness.append((column_name, abs(survived_time_mutated - survived_time_normal)))
    gene_effectiveness.sort(key=lambda _tuple: _tuple[1], reverse=True)

    rank_limit = 10
    effective_genes = [gene_effectiveness[idx][0] for idx in range(rank_limit)]

    # PCA
    generic_alterations = generic_alterations.iloc[:, 1:]
    generic_alterations = generic_alterations.loc[:, effective_genes]
    # clinical_variables = clinical_variables.iloc[:, 1:]
    # generic_alterations = generic_alterations.join(clinical_variables)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(generic_alterations)
    principal_df = pd.DataFrame(data=principal_components, columns=['pc1', 'pc2'])
    principal_df["group"] = group_label

    # print(principal_df)

    # principal axis
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(pd.DataFrame(data=pca.components_).transpose())

    plt.clf()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    groups = [i for i in range(-divisor, divisor + 1) if i != 0]
    for group in groups:
        indices_to_keep = principal_df["group"] == group
        ax.scatter(principal_df.loc[indices_to_keep, 'pc1'], principal_df.loc[indices_to_keep, 'pc2'], s=20)
    ax.legend(groups)
    plt.savefig("output3.png")
