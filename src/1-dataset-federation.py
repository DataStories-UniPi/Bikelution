# %%
import argparse
import pathlib
import pandas as pd

from loguru import logger
from tqdm import tqdm 

import sklearn
sklearn.set_config(transform_output='pandas')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Dataset federation')
    # Arguments related to edge device
    parser.add_argument('--dataset', help='Select dataset', choices=['serveo', 'citi', 'divvy'], type=str, required=True)
    parser.add_argument('--slice', help='Select slice', choices=['train', 'validation', 'test'], type=str, required=True)

    # Prepare parameter dict(s)
    args = parser.parse_args()

    # Define necessary directories
    data_name = args.dataset
    data_dir = pathlib.Path('..')/'data'
    parquet_dir = data_dir/'parquet'

    # Define dataset directory
    dataset = data_dir / data_name / 'h6_w168_multi'
    dataset_slice = args.slice  # train | validation | test
    logger.info(f'{dataset_slice=}')

    # Create necessary directories (if they do not exist)
    (dataset_output_dir := parquet_dir/data_name/dataset_slice).mkdir(parents=True, exist_ok=True)


    # %% [markdown]
    # Load dataset
    df = pd.read_parquet(
        dataset, engine='pyarrow', filters=[('split', '==', dataset_slice)]
    )

    # Display first 5 records
    logger.info(
        df.head()
    )

    logger.info(
        df.groupby(
            level=1 # level 0: split; level 1: station_id; level 2: horizon
        ).apply(
            len
        ).describe()
    )
