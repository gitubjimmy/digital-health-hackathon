from data_loader import get_data
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd


def set_max_survival_time(survival_time_event, max_survival_time):
    for idx in range(len(survival_time_event)):
        if survival_time_event.loc[idx, "event"] == 0:
            survival_time_event.at[idx, "time"] = max_survival_time


def find_effective_genes(survival_time_event, generic_alterations, treatment):
    # effective genes
    gene_effectiveness = []
    for gen_idx in range(300):
        column_name = f"G{gen_idx + 1}"
        mutated_indices_treatment = generic_alterations \
            .loc[(generic_alterations[column_name] == 1) & (treatment["Treatment"] == 1), column_name].index
        normal_indices_treatment = generic_alterations \
            .loc[(generic_alterations[column_name] == 0) & (treatment["Treatment"] == 1), column_name].index
        mutated_indices = generic_alterations \
            .loc[(generic_alterations[column_name] == 1) & (treatment["Treatment"] == 0), column_name].index
        normal_indices = generic_alterations \
            .loc[(generic_alterations[column_name] == 0) & (treatment["Treatment"] == 0), column_name].index

        survived_time_mutated_treatment = survival_time_event.loc[mutated_indices_treatment, "time"].mean()
        survived_time_normal_treatment = survival_time_event.loc[normal_indices_treatment, "time"].mean()
        survived_time_mutated = survival_time_event.loc[mutated_indices, "time"].mean()
        survived_time_normal = survival_time_event.loc[normal_indices, "time"].mean()

        t1 = survived_time_mutated_treatment - survived_time_normal_treatment
        t2 = survived_time_mutated - survived_time_normal

        gene_effectiveness.append((column_name, t1 - t2))
    gene_effectiveness.sort(key=lambda _tuple: _tuple[1], reverse=True)
    return gene_effectiveness


def print_effective_genes(gene_effectiveness):
    print('\ngene_effectiveness')
    for effective_gene in gene_effectiveness:
        print(effective_gene)


def main():
    # get data
    clinical_variables, generic_alterations, survival_time_event, treatment = get_data()
    n_data = treatment.shape[0]  # row size
    n_genes = generic_alterations.shape[1] - 1  # gene count
    set_max_survival_time(survival_time_event, 120)  # set survival time of alive people

    # get histogram of survival_time
    hist = survival_time_event.hist(column="time", range=(0, 200), bins=10)[0][0]
    hist.set_xlabel("survived time")
    hist.set_ylabel("# of people")
    fig = hist.get_figure()
    fig.savefig("outputs/survival_time_hist.png")

    # group people by their survived time & treatment
    sorted_survival_time_event = survival_time_event.sort_values(by="time")
    n_groups = 1
    group_label = [0 for _ in range(n_data)]
    for k in range(n_groups):
        start = int(n_data / n_groups * k)
        end = int(n_data / n_groups * (k + 1))
        for idx in range(start, end):
            had_treatment = treatment.loc[idx, "Treatment"]
            group_label[sorted_survival_time_event.iloc[idx, 0]] = f'{k}_treatment' if had_treatment else k
            # higher k <=> survived longer

    # visualize
    group_label_pd = pd.Series(group_label, name='frequency')
    group_label_pd = group_label_pd.value_counts()
    with open('outputs/group_label_count.md', 'w') as group_label_count_md:
        group_label_count_md.write("## group frequency table  \n")
        group_label_count_md.write(group_label_pd.to_frame().to_markdown())

    # gene relativity graph
    relativity_graph = [[0 for _ in range(n_genes)] for _ in range(n_genes)]
    relativity_graph_xor = [[0 for _ in range(n_genes)] for _ in range(n_genes)]
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

    # extra stats from relativity graph
    print('average_mutation_per_person', sum(mutation_per_person) / len(mutation_per_person))
    flat_relativity_graph_xor = [cell_value for row in relativity_graph_xor for cell_value in row if cell_value > 0]
    print('min value in relativity_graph_xor', min(flat_relativity_graph_xor))

    # find effective genes
    n_desired_genes = 50
    gene_effectiveness_max120 = \
        find_effective_genes(survival_time_event, generic_alterations, treatment)[:n_desired_genes]
    print_effective_genes(gene_effectiveness_max120[:10])
    print('...\n')

    set_max_survival_time(survival_time_event, 999)
    gene_effectiveness_max999 = \
        find_effective_genes(survival_time_event, generic_alterations, treatment)[:n_desired_genes]
    print_effective_genes(gene_effectiveness_max999[:10])
    print('...')

    effective_genes_max120 = [_tuple[0] for _tuple in gene_effectiveness_max120]
    effective_genes_max999 = [_tuple[0] for _tuple in gene_effectiveness_max999]
    effective_genes_intersection = list(set(effective_genes_max120) & set(effective_genes_max999))
    print('\neffective_genes_intersection')
    print(', '.join(effective_genes_intersection[:10] + ['...']))
    print('count', len(effective_genes_intersection))

    # effective_genes = [gene_effectiveness[idx][0] for idx in range(rank_limit)]
    # effective_genes = [54, 238, 11, 259, 234, 223, 158]
    # effective_genes = [f'G{gene_number}' for gene_number in effective_genes]
    effective_genes = effective_genes_intersection

    #######
    # PCA #
    #######

    # remove alive data
    generic_alterations = generic_alterations.loc[survival_time_event["event"] == 1]
    clinical_variables = clinical_variables.loc[survival_time_event["event"] == 1]
    treatment = treatment.loc[survival_time_event["event"] == 1]
    group_label = [group_label[idx] for idx in range(len(group_label)) if idx in treatment.index]
    survival_time_event = survival_time_event.loc[survival_time_event["event"] == 1]

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
    groups = set(group_label)
    for group in groups:
        indices_to_keep = principal_df["group"] == group
        ax.scatter(principal_df.loc[indices_to_keep, 'pc1'], principal_df.loc[indices_to_keep, 'pc2'], s=20)
    ax.legend(groups)
    plt.savefig("outputs/pca.png")


if __name__ == '__main__':
    main()
