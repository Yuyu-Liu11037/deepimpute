# DeepImpute: an accurate and efficient deep learning method for single-cell RNA-seq data imputation
### Enviroment
```
conda env create -f environment.yml
```

### Usage
```
usage: deepImpute [-h] [-o OUTPUT] [--cores CORES]
                  [--cell-axis {rows,columns}] [--limit LIMIT]
                  [--minVMR MINVMR] [--subset SUBSET]
                  [--learning-rate LEARNING_RATE] [--batch-size BATCH_SIZE]
                  [--max-epochs MAX_EPOCHS] [--hidden-neurons HIDDEN_NEURONS]
                  [--dropout-rate DROPOUT_RATE]
                  [--output-neurons OUTPUT_NEURONS] [--n_pred N_PRED]
                  [--policy POLICY]
                  inputFile

scRNA-seq data imputation using DeepImpute.

positional arguments:
  inputFile             Path to input data.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Path to output data counts. Default: ./imputed.csv
  --cores CORES         Number of cores. Default: all available cores
  --cell-axis {rows,columns}
                        Cell dimension in the matrix. Default: rows
  --limit LIMIT         Genes to impute (e.g. first 2000 genes). Default: auto
  --minVMR MINVMR       Min Variance over mean ratio for gene exclusion. Gene
                        with a VMR below ${minVMR} are discarded. Used if
                        --limit is set to 'auto'. Default: 0.5
  --subset SUBSET       Cell subset to speed up training. Either a ratio
                        (0<x<1) or a cell number (int). Default: 1 (all)
  --learning-rate LEARNING_RATE
                        Learning rate. Default: 0.0001
  --batch-size BATCH_SIZE
                        Batch size. Default: 64
  --max-epochs MAX_EPOCHS
                        Maximum number of epochs. Default: 500
  --hidden-neurons HIDDEN_NEURONS
                        Number of neurons in the hidden dense layer. Default:
                        256
  --dropout-rate DROPOUT_RATE
                        Dropout rate for the hidden dropout layer (0<rate<1).
                        Default: 0.2
  --output-neurons OUTPUT_NEURONS
                        Number of output neurons per sub-network. Default: 512
  --n_pred N_PRED       Number of predictors to consider. Consider using this
                        parameter if your RAM is limited or if you have a high
                        number of features. Default: All genes with nonzero
                        VMR
  --policy POLICY       Whether to restore positive values from the raw
                        dataset or keep the max between the imputed values and
                        the raw values. Choices are ['restore', 'max'].
                        Default: restore
```

Python:

```python
python -m deepimpute.deepImpute /workspace/deepimpute/data/citeseq_processed.csv --output ./data/citese
q_imputed.csv --limit 2000
```
### Remark
The authors used log1p and reversal in their own code, which is confusing (when calculating pcc, mae, rmse: notice comparing raw data or log-transformed data).