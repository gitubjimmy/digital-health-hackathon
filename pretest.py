def pretest():

    import pandas as pd
    from data_prep_utils import get_processed_data

    samples = get_processed_data().samples
    gene_idx = tuple(map(lambda key: 'G{}'.format(key), range(1, 300 + 1)))

    tau1_col = []
    tau2_col = []
    ts_col = []

    for gene in gene_idx:

        df = samples[[gene, 'Treatment', 'time']]
        unique_df = df[[gene, 'Treatment']].drop_duplicates()

        if len(unique_df[unique_df.Treatment == 1]) == 2:  # tau1
            looking = df[df.Treatment == 1]
            tau1 = abs(looking[looking[gene] == 1].time.mean() - looking[looking[gene] == 0].time.mean())
        else:
            tau1 = float('nan')

        if len(unique_df[unique_df.Treatment == 0]) == 2:  # tau1
            looking = df[df.Treatment == 0]
            tau2 = abs(looking[looking[gene] == 1].time.mean() - looking[looking[gene] == 0].time.mean())
        else:
            tau2 = float('nan')

        tau_score = abs(tau1 - tau2)

        tau1_col.append(tau1)
        tau2_col.append(tau2)
        ts_col.append(tau_score)

    df = pd.DataFrame({"tau1": tau1_col, "tau2": tau2_col, "abs(t1-t2)": ts_col}, index=gene_idx)

    print("# Raw Data Tau Score  \n")
    print("## Sort by tau1\n")
    print(df.sort_values(by="tau1", ascending=False).to_markdown())
    print()
    print("## Sort by tau2\n")
    print(df.sort_values(by="tau2", ascending=False).to_markdown())
    print()
    print("## Sort by abs(tau1-tau2)\n")
    print(df.sort_values(by="abs(t1-t2)", ascending=False).to_markdown())
    print()


if __name__ == '__main__':
    pretest()
