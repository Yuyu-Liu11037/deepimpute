import pandas as pd

from deepimpute.parser import parse_args
from deepimpute.multinet import MultiNet
from tqdm import tqdm


chunk_size = 10000


def deepImpute(**kwargs):

    args = parse_args()

    for key, value in kwargs.items():
        setattr(args, key, value)

    with open(args.inputFile) as f:
        total_lines = sum(1 for line in f) - 1

    print('reading data')
    data = pd.DataFrame()
    with tqdm(total=total_lines, desc="Reading CSV") as pbar:
        for chunk in pd.read_csv(args.inputFile, chunksize=chunk_size, index_col=0):
            data = pd.concat([data, chunk])
            pbar.update(chunk_size)

    if args.cell_axis == "columns":
        data = data.T

    NN_params = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'ncores': args.cores,
        'sub_outputdim': args.output_neurons,
        'architecture': [
            {"type": "dense", "activation": "relu", "neurons": args.hidden_neurons},
            {"type": "dropout", "activation": "dropout", "rate": args.dropout_rate}]
    }

    multi = MultiNet(**NN_params)
    print('train network')
    multi.fit(data, NN_lim=args.limit, cell_subset=args.subset, minVMR=args.minVMR, n_pred=args.n_pred)

    print('start impute')
    imputed = multi.predict(data, imputed_only=False, policy=args.policy)

    if args.output is not None:
        with open(args.output, 'w') as f:
            imputed.iloc[:0].to_csv(f, index=True)
            with tqdm(total=len(imputed), desc="Writing CSV") as pbar:
                for i in range(0, len(imputed), chunk_size):
                    imputed.iloc[i:i + chunk_size].to_csv(f, header=False, index=True)
                    pbar.update(chunk_size)
    else:
        return imputed

if __name__ == "__main__":
    deepImpute()
