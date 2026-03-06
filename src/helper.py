# Essentials / Python natives
import re
import json
from math import ceil
from collections import OrderedDict

# Data Management
import numpy as np
import pandas as pd

# Scikit-learn/-time
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError

# ML libraries
import xgboost as xgb
import torch

# Logging
import logging
from loguru import logger
from tqdm import tqdm


class InterceptHandler(logging.Handler):
    """
    Forward a standard LogRecord to loguru, preserving level, message
    and exception information.
    """
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:                 # custom levels fallback
            level = record.levelno

        logger.opt(
            depth=6,                       # correct call‑site information
            exception=record.exc_info
        ).log(level, record.getMessage())


def mem_report():
    import psutil, os 

    proc = psutil.Process(os.getpid())
    return (
        f"RSS={proc.memory_info().rss / 1024**2:.1f} MiB "
        # f" ArrowAllocated={pyarrow.default_memory_pool().bytes_allocated() / 1024**2:.1f} MiB"
    )


def extract_numbers(filename):
    # Sort by extracting round and epoch numbers
    round_match = re.search(r'round(\d+)', filename)
    epoch_match = re.search(r'epoch(\d+)', filename)
    
    round_num = int(round_match.group(1)) if round_match else 0
    epoch_num = int(epoch_match.group(1)) if epoch_match else 0
    
    return (round_num, epoch_num)


def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def weighted_sum(results, metric):
    # Weigh accuracy of each client by number of examples used
    results_metrics_aggregated, num_examples_aggregated = [], []

    for _, result in results:        
        # If the metric does not exist in the metrics dict sent by the client, 
        # proceed with the next client in the list
        if metric not in result.metrics:
            logger.warning(f'[strategy][weighted_sum] Result does not contain {metric} metric!')
            continue

        # Fetch ```metric``` value
        result_metric = result.metrics[metric]

        # If the ```metric``` is string (i.e., jsonified list), deserialize the value
        if type(result_metric) is str:
            result_metric = np.array(json.loads(result_metric))

        # Finally, weight the result metric
        results_metrics_aggregated.append(result_metric * result.num_examples)
        num_examples_aggregated.append(result.num_examples)

    # Perform weighted sum and return result
    return np.sum(results_metrics_aggregated, axis=0) / np.sum(num_examples_aggregated, axis=0)


def smape(y_true, y_pred, *args, **kwargs):
    '''
        Calculate Symmetric Mean Absolute Percentage Error (sMAPE)

        Parameters:
        y_true (array-like): Array of actual values
        y_pred (array-like): Array of forecasted values

        Returns:
        float: sMAPE value (%)
    '''
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    # Calculate the sMAPE
    smape_value = np.mean(numerator / (denominator + 1e-9), axis=0)
    
    return smape_value


def wape(y_true, y_pred, *args, **kwargs):
    '''
        Calculate Weighted Absolute Percentage Error (WAPE)
        Source: https://medium.com/@vinitkothari.24/time-series-evaluation-metrics-mape-vs-wmape-vs-smape-which-one-to-use-why-and-when-part1-32d3852b4779

        Parameters:
        y_true (array-like): Array of actual values
        y_pred (array-like): Array of forecasted values

        Returns:
        float: WAPE value (%)
    '''

    numerator = np.abs(y_true - y_pred).sum(axis=0)
    denominator = y_true.sum(axis=0)

    # Calculate the WAPE
    wape_value = numerator / (denominator + 1e-9)
    return wape_value


def model_score(predictions, metric_fun, **kwargs):
    eps = kwargs.pop('eps', 0)
    y_true_column = predictions.columns[0]  # In the method the actual value is expected to be given first

    return pd.Series(
        data=[
            metric_fun(
                predictions[y_true_column].values + eps, 
                predictions[column].values + eps, 
                **kwargs
            ) for column in predictions.columns[1:]
        ],
        index=predictions.columns[1:]
    )


def evaluate_predictions(predictions, y_true_name, y_pred_names, eval_funs):
    models_to_compare = [y_true_name, *y_pred_names]
    eval_funs_res = {}

    for i, (fun_name, fun, fun_kwargs) in enumerate(eval_funs):
        
        if not isinstance(fun, MeanAbsoluteScaledError):
            eval_funs_res[f'{i}_{fun_name}'] = predictions.loc[:, models_to_compare].groupby(level=0).apply(
                lambda l: model_score(l, fun, **fun_kwargs)
            ) 
            continue

        y_train = fun_kwargs.pop('y_train', None)
        oid_indices = fun_kwargs.pop('oid_indices', None)

        eval_funs_res[f'{i}_{fun_name}'] = predictions.loc[:, models_to_compare].groupby(level=0).apply(
            lambda l: model_score(l, fun, **{
                'y_train': y_train[oid_indices[l.name]].values,
                **fun_kwargs
            })
        ) 
    
    return pd.concat(eval_funs_res)


def ebike_load_data_xgboost(dataset_name, dataset_params, oid, dataset_slice=['train', 'validation', 'test']):
    # data_template = f'{dataset_name}_{{0}}_h1_w24_v1_client{{1}}.parquet.gzip'
    # data_template = f'{dataset_name}_{{0}}_h6_w168_v1_client{{1}}.parquet.gzip'
    data_template = f'{dataset_name}_{{0}}_h6_w168_v2_client{{1}}.parquet.gzip'
    
    # Read train dataset
    if 'train' in dataset_slice:
        df_train = pd.read_parquet(
            data_template_train := dataset_params['parquet_dir']/dataset_name/'train'/data_template.format('train', oid), 
            engine='fastparquet'
        )
        # df_train.dropna(inplace=True)
        logger.info(f'[ebike_load_data_xgboost] Loaded training dataset from f{data_template_train}')
    else:
        df_train = None

    # Read validation dataset
    if 'validation' in dataset_slice:
        df_dev = pd.read_parquet(
            data_template_dev := dataset_params['parquet_dir']/dataset_name/'validation'/data_template.format('validation', oid), 
            engine='fastparquet'
        )
        # df_dev.dropna(inplace=True)
        logger.info(f'[ebike_load_data_xgboost] Loaded validation dataset from f{data_template_dev}')
    else:
        df_dev = None

    # Read test dataset
    if 'test' in dataset_slice:
        df_test = pd.read_parquet(
            data_template_test := dataset_params['parquet_dir']/dataset_name/'test'/data_template.format('test', oid), 
            engine='fastparquet'
        )
        # df_test.dropna(inplace=True)
        logger.info(f'[ebike_load_data_xgboost] Loaded test dataset from f{data_template_test}')
    else:
        df_test = None

    X_train, X_dev, X_test = (
        df_train.drop(dataset_params['y_feats'], axis=1) if df_train is not None else None,
        df_dev.drop(dataset_params['y_feats'], axis=1) if df_dev is not None else None,
        df_test.drop(dataset_params['y_feats'], axis=1) if df_test is not None else None
    )

    y_train, y_dev, y_test = (
        df_train.loc[:, dataset_params['y_feats']] if df_train is not None else None,
        df_dev.loc[:, dataset_params['y_feats']] if df_dev is not None else None,
        df_test.loc[:, dataset_params['y_feats']] if df_test is not None else None
    )

    logger.info(
        f'[ebike_load_data_xgboost] Training samples: {len(X_train) if df_train is not None else "-"} | '
        f'Validation samples: {len(X_dev) if df_dev is not None else "-"} | '
        f'Testing samples: {len(X_test) if df_test is not None else "-"}'
    )

    return (
        {'X': X_train, 'y': y_train} if df_train is not None else None, # Train set
        {'X': X_dev, 'y': y_dev} if df_dev is not None else None, # Dev set
        {'X': X_test, 'y': y_test} if df_test is not None else None, # Test set
    )


def xgb_tree_predictions(xgb_model, X, **kwargs):
    booster = xgb_model.get_booster()
    n_estimators = booster.num_boosted_rounds() # Get the number of estimators

    X_dm = xgb.DMatrix(X, **kwargs)

    tree_preds = []
    for tree_ix in range(n_estimators):
        tree_pred = booster.predict(
            X_dm, 
            iteration_range=(tree_ix, tree_ix+1), 
            output_margin=True
        )   # <n_samples, n_outputs>
        
        tree_preds.extend(
            [tree_pred[:, i] for i in range(tree_pred.shape[1])]
        )   # List[NDarray[n_samples,]]

    return tree_preds  # List[NDarray[n_samples, n_outputs]]


def arr_iteration(df, batch_size):
    for ix in range(0, len(df), batch_size):
        yield ix, df.iloc[ix:ix+batch_size, :]


def fedxgbllr_cnn_create_parquet_dataset(aggregated_trees, dataset, dataset_params, batch_size, **kwargs):
    import gc
    import fastparquet as fp

    # Create the Parquet schema for the output
    pq_schema_X = [
        f'tree_{client_ix}_estimator_{estimator_ix}_output_{output_ix}'
        for client_ix in range(len(aggregated_trees)) #clients
        for estimator_ix in range(dataset_params['trees_per_client']) #estimators
        for output_ix in range(dataset_params['in_channels']) #outputs
    ]

    pq_schema_y = [
        f'output_{output_ix}'
        for output_ix in range(dataset_params['in_channels']) #outputs
    ]

    # Filename template of the output Parquet files
    pq_out_filename = f'cnn_{dataset_params["dataset_slice"]}_dataset__'+\
                      f'n_estimators_{dataset_params["trees_per_client"]}'+\
                      f'.{{0}}.client{dataset_params["oid"]}.parquet.gzip'
    
    save_path = (
        dataset_params['save_path'].parent/
        dataset_params['dataset_name']
    )
    dataset_X_path = save_path / pq_out_filename.format('X')
    dataset_y_path = save_path / pq_out_filename.format('y')

    logger.info(f'Saving FedXGBllr {dataset_params["dataset_slice"]} dataset to \n\t{dataset_X_path} \n\t{dataset_y_path}\n')
    if dataset_X_path.exists() and dataset_y_path.exists():
        logger.info(f'The files already exist! Skipping...')
        return dataset_X_path, dataset_y_path

    # Process samples in batches
    for ix, arr_batch in tqdm(
        arr_iteration(dataset['X'], batch_size), 
        total=ceil(len(dataset['X']) / batch_size), 
        desc=f'[client #{dataset_params["oid"]}] {dataset_params["dataset_slice"]} dataset features'
    ):
        # logger.debug(f'[client #{dataset_params["oid"]}] Memory report - loop start - {mem_report()}')
        arr_batch_pred = []

        # Get the predictions for each tree in the federation
        for client_id_tree, client_id in aggregated_trees:
            # logger.debug(f'[client #{dataset_params["oid"]}] Getting estimator predictions from tree #{client_id}...')
            arr_batch_pred.extend(
                xgb_tree_predictions(
                    xgb_model=client_id_tree, 
                    X=arr_batch,
                    **kwargs
                )
            )
    
        # Convert the list of Arrow arrays into one RecordBatch
        batch = pd.DataFrame(
            {name: col for name, col in zip(pq_schema_X, arr_batch_pred)}
        )
        fp.write(
            str(dataset_X_path),
            batch,
            write_index=False,
            compression='GZIP',
            append=ix!=0
        )

        del arr_batch_pred, batch
        gc.collect()
        # logger.debug(f'[client #{dataset_params["oid"]}] Memory report - loop end - {mem_report()}')

    # Convert the list of Arrow arrays into one RecordBatch
    for ix, arr_batch in tqdm(
        arr_iteration(dataset['y'], batch_size), 
        total=ceil(len(dataset['y']) / batch_size), 
        desc=f'[client #{dataset_params["oid"]}] {dataset_params["dataset_slice"]} dataset labels'
    ):
        batch = pd.DataFrame(
            {
                name: col 
                for name, col 
                in zip(
                    pq_schema_y, 
                    [
                        arr_batch.values[:, i] for i in range(arr_batch.shape[1])
                    ]
                )
            }
        )

        fp.write(
            str(dataset_y_path),
            batch,
            write_index=False,
            compression='GZIP',
            append=ix!=0
        )

        del batch
        gc.collect()

    return dataset_X_path, dataset_y_path


def fedxgbllr_cnn_create_tensor_dataset(aggregated_trees, dataset, dataset_params, **kwargs):
    logger.debug(f'Memory report - function start - {mem_report()}')
    dataset_X_path, dataset_y_path = fedxgbllr_cnn_create_parquet_dataset(
        aggregated_trees, 
        dataset, 
        dataset_params, 
        batch_size=dataset_params['parquet_bs'], 
        **kwargs
    )
    logger.debug(f'Memory report - function end - {mem_report()}')
    logger.info(f'Loading FedXGBllr {dataset_params["dataset_slice"]} dataset from \n\t {dataset_X_path} \n\t{dataset_y_path}\n')
    
    cnn_dataset_params = dict(
        X_path=dataset_X_path,
        y_path=dataset_y_path,
        trees_per_client = dataset_params['trees_per_client'],
        num_clients = len(aggregated_trees),
        in_channels = dataset_params['in_channels'],
        batch_size = dataset_params['bs'],
        shuffle = dataset_params['dataset_slice'] == 'train'
    )
    logger.debug(f'Torch dataset parameters: {cnn_dataset_params}')

    return cnn_dataset_params
