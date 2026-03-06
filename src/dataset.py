import numpy as np
import pyarrow.parquet as pq

import torch
from torch.utils.data import IterableDataset

from pathlib import Path
from loguru import logger

class FedXGBllrDataset(IterableDataset):
    """
    Iterable dataset that streams tree-level predictions from a distributed
    XGBoost federation. Each row-group of the Parquet files is loaded lazily,
    shuffled (optionally), converted to a float32 tensor, and yielded in
    mini-batches.

    Parameters
    ----------
    X_path : str
        Path to the Parquet file containing the concatenated tree outputs.
    y_path : str
        Path to the Parquet file containing the labels.
    trees_per_client : int
        Number of trees each client contributes.
    num_clients : int
        Total number of clients (or total number of trees if you prefer).
    in_channels : int
        Number of output channels per tree (i.e., dimensionality of a tree
        prediction).
    batch_size : int, default 64
        Mini-batch size.
    shuffle : bool, default True
        Shuffle the order of row-groups *and* rows within each group.
    """

    def __init__(
        self,
        X_path: Path,
        y_path: Path,
        trees_per_client: int,
        num_clients: int,
        in_channels: int,
        batch_size: int = 64,
        shuffle: bool = True
    ):
        self.X_path = X_path
        assert self.X_path.exists(), f"{self.X_path=} does not exist"
        
        self.y_path = y_path
        assert self.y_path.exists(), f"{self.y_path=} does not exist"

        (
            self.in_channels,
            self.batch_size,
            self.shuffle
        ) = (
            in_channels,
            batch_size,
            shuffle
        )

        # Pre-compute total number of trees
        self.total_trees = trees_per_client * num_clients
        
        try:
            # Create pointer to Parquet files
            X_pq = pq.ParquetFile(
                self.X_path,
                memory_map=False,
            )
            y_pq = pq.ParquetFile(
                self.y_path,
                memory_map=False
            )

            self.num_row_groups = X_pq.num_row_groups
            assert self.num_row_groups == y_pq.num_row_groups, "Parquet files (X, y) don't match in length!"

            # Get total number of records
            self._total_samples = sum(
                X_pq.metadata.row_group(i).num_rows
                for i in range(self.num_row_groups)
            )
        finally:
            # Release allocated memory
            X_pq.close()
            y_pq.close()

    def __len__(self):
        return (self._total_samples // self.batch_size) + 1

    def __iter__(self):
        row_group_ix = np.arange(self.num_row_groups)

        # Shuffle row group indices if shuffle is enabled
        if self.shuffle:
            np.random.shuffle(row_group_ix)
            logger.debug(f'Current (row group) permutation (first 32 indices): {row_group_ix[:32]}')

        try:
            # Create pointer to Parquet files
            X_pq = pq.ParquetFile(
                self.X_path,
                memory_map=False,
            )
            y_pq = pq.ParquetFile(
                self.y_path,
                memory_map=False
            )

            for ix in row_group_ix:
                # Convert to NumPy array (efficient; no pandas)
                X = X_pq.read_row_group(ix)
                y = y_pq.read_row_group(ix)

                X = np.column_stack([col.to_numpy() for col in X.columns])
                y = np.column_stack([col.to_numpy() for col in y.columns])

                # Shuffle within row group if enabled
                if self.shuffle:
                    perm = np.random.permutation(X.shape[0])
                    X, y = (
                        X[perm], 
                        y[perm]
                    )

                # Convert to tensor with shape: [records, estimators, features]
                X_tensor = torch.from_numpy(
                    # X.values
                    X
                ).view(
                    -1,  # number of records
                    self.total_trees,  # total number of estimators
                    self.in_channels  # number of features per estimator
                ).swapaxes(
                    -2,
                    -1
                )  # shape: [records, features, estimators]

                y_tensor = torch.from_numpy(
                    y
                )  # shape: [records, features]
                
                # Yield batches of data
                for i in range(0, X_tensor.shape[0], self.batch_size):                    
                    xb, yb = (
                        X_tensor[i:i + self.batch_size],
                        y_tensor[i:i + self.batch_size]
                    )

                    yield (xb, yb)
        finally:
            # Release allocated memory
            X_pq.close()
            y_pq.close()
