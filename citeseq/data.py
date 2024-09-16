import anndata as ad
import pandas as pd
import scanpy as sc
import numpy as np
import scipy.sparse as sp
import sys


SITE1_CELL = 16311
SITE2_CELL = 25171
SITE3_CELL = 32029
SITE4_CELL = 16750
GEX = 2000


def reverse_log1p_sparse(sparse_matrix):
    return sp.csr_matrix(np.expm1(sparse_matrix.toarray()))


adata = ad.read_h5ad('/workspace/deepimpute/data/citeseq_processed.h5ad')
adata.var_names_make_unique()

adata_GEX = adata[:, adata.var["feature_types"] == "GEX"].copy()
adata_ADT = adata[:, adata.var["feature_types"] == "ADT"].copy()
### step 1
sc.pp.normalize_total(adata_GEX, target_sum=1e4)
sc.pp.normalize_total(adata_ADT, target_sum=1e4)
### step 2
sc.pp.log1p(adata_GEX)
sc.pp.log1p(adata_ADT)
### step 3
sc.pp.highly_variable_genes(
    adata_GEX,
    n_top_genes=2000,
    subset=True
)
adata = ad.concat([adata_GEX, adata_ADT], axis=1, merge="first")
adata.write_h5ad('/workspace/deepimpute/data/citeseq_preprocessed.h5ad')
adata[:, adata.var["feature_types"] == "GEX"].X = reverse_log1p_sparse(adata[:, adata.var["feature_types"] == "GEX"].X)
adata[:, adata.var["feature_types"] == "ADT"].X = reverse_log1p_sparse(adata[:, adata.var["feature_types"] == "ADT"].X)

adata.obs.to_csv('/workspace/deepimpute/data/citeseq_processed_cell_metadata.csv')
adata.var.to_csv('/workspace/deepimpute/data/citeseq_processed_gene_metadata.csv')

X = adata.X.toarray()
X[SITE1_CELL + SITE2_CELL: SITE1_CELL + SITE2_CELL + SITE3_CELL, :GEX] = 0

data = pd.DataFrame(X, index=adata.obs.index, columns=adata.var.index)

chunk_size = 10000

print(f'Start writing.\n')
with open('/workspace/deepimpute/data/citeseq_processed.csv', 'w') as f:
    data.iloc[:0].to_csv(f, index=True)
    
    total_rows = len(data)
    for i in range(0, total_rows, chunk_size):
        data.iloc[i:i + chunk_size].to_csv(f, header=False, index=True)
        print(f'Written {i + chunk_size if i + chunk_size < total_rows else total_rows}/{total_rows} rows')

print("Finished writing the CSV file.")
