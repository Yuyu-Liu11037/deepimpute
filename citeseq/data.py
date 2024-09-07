import anndata as ad
import pandas as pd
import scanpy as sc
import sys


SITE1_CELL = 16311
SITE2_CELL = 25171
SITE3_CELL = 32029
SITE4_CELL = 16750
GEX = 13953


adata = ad.read_h5ad('/workspace/deepimpute/data/citeseq_processed.h5ad')
adata.var_names_make_unique()
adata.obs.to_csv('/workspace/deepimpute/data/citeseq_processed_cell_metadata.csv')
adata.var.to_csv('/workspace/deepimpute/data/citeseq_processed_gene_metadata.csv')
sys.exit()

X = adata.X.toarray()
X[SITE1_CELL + SITE2_CELL: SITE1_CELL + SITE2_CELL + SITE3_CELL, :GEX] = 0

data = pd.DataFrame(X, index=adata.obs.index, columns=adata.var.index)

chunk_size = 1000  # 每次写入 1000 行

print(f'Start writing.\n')
with open('/workspace/deepimpute/data/citeseq_processed.csv', 'w') as f:
    data.iloc[:0].to_csv(f, index=True)
    
    total_rows = len(data)
    for i in range(0, total_rows, chunk_size):
        data.iloc[i:i + chunk_size].to_csv(f, header=False, index=True)
        print(f'Written {i + chunk_size if i + chunk_size < total_rows else total_rows}/{total_rows} rows')

print("Finished writing the CSV file.")
