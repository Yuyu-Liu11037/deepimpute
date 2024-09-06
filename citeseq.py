import anndata as ad
import pandas as pd


from deepimpute.multinet import MultiNet


adata = ad.read_h5ad('/workspace/deepimpute/data/citeseq_processed.h5ad')

# 将表达矩阵 (X) 转换为 DataFrame 并保存为 CSV
df = pd.DataFrame(adata.X, index=adata.obs.index, columns=adata.var.index)
df.to_csv('/workspace/deepimpute/data/citeseq_processed.csv')
adata.obs.to_csv('/workspace/deepimpute/data/citeseq_processed_cell_metadata.csv')
adata.var.to_csv('/workspace/deepimpute/data/citeseq_processed_gene_metadata.csv')
