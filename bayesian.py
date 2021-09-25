from data_loader import get_data
from utils import write_to_csv
import numpy as np

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
    print(L)
    print(k)
    for i in range(300):
        for j in range(300):
            W[i][j] = (L[i][j] + A[i][j]) / (min(k[i], k[j]) + 1 - A[i][j])
            if i != j:
                W[i][j] *= 1000
    write_to_csv('output.csv', W)
