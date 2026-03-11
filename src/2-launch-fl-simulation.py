import os
os.environ['ARROW_DEFAULT_MEMORY_POOL']='jemalloc'
os.environ['ARROW_MAX_THREADS']='1'
os.environ['TQDM_DISABLE']='1'

import json
import pathlib
import argparse
import random
from datetime import datetime

import numpy as np 
import torch

# Federated Learning
import flwr as fl
from flwr.client import (
    Client
)
from flwr.common import (
    Context
)

from xgb_client import FedXGBllr_Client
from helper import InterceptHandler, smape, wape
from train import evaluate_model_multihead
from strategy import FedXGBllr

# Log handling
from loguru import logger
from flwr.common.logger import logger as flwr_logger

# Flower logs 
flwr_logger.handlers.clear()    
flwr_logger.propagate = False
flwr_logger.addHandler(InterceptHandler())

# Example: coloured console + rotating file, both using the same format
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# Define global variables
CFG_DATA_DIR = pathlib.Path('..')/'data'
CFG_FIGURES_DIR = CFG_DATA_DIR/'fig'
CFG_LOGFILE_DIR = CFG_DATA_DIR/'logs'
CFG_PARQUET_DIR = CFG_DATA_DIR/'parquet'
CFG_PICKLE_DIR = CFG_DATA_DIR/'pickle'
with (
    open(CFG_DATA_DIR / 'txt' / 'client_ids__serveo.txt', 'r') as fp_serveo, 
    open(CFG_DATA_DIR / 'txt' / 'client_ids__divvy.txt', 'r') as fp_divvy, 
    open(CFG_DATA_DIR / 'txt' / 'client_ids__citi.txt', 'r') as fp_citi, 
):
    CFG_CLIENT_IDS = {
        # Identifiers of Serveo dataset
        'serveo': json.load(fp_serveo),
        # Identifiers of Divvy dataset
        'divvy': json.load(fp_divvy),
        # Identifiers of Citi dataset
        'citi': json.load(fp_citi)
    }
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='FedXGB Local Worker')
    # Arguments related to edge device
    parser.add_argument('--federation', help='Select Federated Dataset', choices=['serveo', 'divvy', 'citi'], type=str, required=True)
    # parser.add_argument('--oid', help='Object ID', type=str, required=True)
    # # XGBOOST PARAMS
    parser.add_argument('--n_estimators', help='Number of trees (estimators) in the XGBoost model', type=int, default=27)
    # parser.add_argument('--max_depth', help='Maximum depth of a tree', type=int, default=6)
    # # DATA PARAMS
    parser.add_argument('--parquet_bs', help='Parquet processing batch size', default=4096, type=int)
    parser.add_argument('--bs', help='Dataset Batch Size', default=32, type=int)
    # # FL SIMULATION PARAMS
    parser.add_argument('--num_rounds', help='#FL Rounds (default: 30)', default=30, type=int, required=False)
    parser.add_argument('--local_epochs', help='#Local Epochs (default: 3)', default=3, type=int, required=False)
    parser.add_argument('--early_stop', help='Enable Early Stopping mechanism during edge devices\' training', action="store_true")
    parser.add_argument('--patience', help='Patience (#Epochs) for Early Stopping (default: 3)', default=3, type=int)
    parser.add_argument('--conv_channels', help='FedXGBllr 1D-CNN - no. of convolutional channels', default=8, required=False, type=int)
    parser.add_argument('--fc_layers', help='FedXGBllr 1D-CNN - Number/Size of FC Layers', type=str, default='')
    parser.add_argument('--dropout_rate', help='FedXGBllr 1D-CNN - dropout probability', default=0.5, required=False, type=float)
    parser.add_argument('--mu', help='Proximal $\mu$', default=0.01, type=float, required=False)
    parser.add_argument('--fraction_fit', help='#clients to train per round (%%)', default=1.0, type=float, required=False)
    parser.add_argument('--fraction_eval', help='#clients to evaluate per round (%%)', default=1.0, type=float, required=False)

    # Prepare parameter dict(s)
    args = parser.parse_args()

    # Ensure reproducible client selection 
    random.seed(random_state__server := 42)
    np.random.seed(random_state__server)

    # Related to Clients' initialization
    client_ids = {
        str(k):v for k, v in enumerate(CFG_CLIENT_IDS[args.federation])
    }
    
    if args.federation == 'serveo':
        dataset_params__y_feats = ['target_demand_inbound', 'target_demand_outbound']
    else:
        dataset_params__y_feats = ['target_demand_arrivals', 'target_demand_departures']
    
    dataset_params = dict(
        dataset_tag=args.federation,
        njobs=1,
        data_dir=CFG_DATA_DIR,
        figures_dir=CFG_FIGURES_DIR,
        parquet_dir=CFG_PARQUET_DIR,
        pickle_dir=CFG_PICKLE_DIR,
        y_feats=dataset_params__y_feats,
        bs=args.bs,
        parquet_bs=args.parquet_bs
    )
    logger.info(f'Dataset parameters: {dataset_params}')

    def client_fn(context: Context) -> Client:
        """Return a client instance for the given client ID"""
        # Configure Loguru inside the actor
        client_id = client_ids[context.node_config['partition-id']]
        
        # Add file handler with thread-safe enqueue
        logger.add(
            CFG_LOGFILE_DIR / f'{args.federation}_flwr_simulation' / f"fedxgb_client{client_id}.log",
            rotation="10 MB",
            retention="7 days",
            enqueue=True,  # Critical for thread safety
            backtrace=True,
            mode="a",
            level="DEBUG"
        )

        # Set random seed for Hyperparameter selection
        # NOTE: Ensure diverse XGBoost models, as well as reproducible results...
        random_state = np.mod(
            hash(client_id), 
            2**32-1
        )
        np.random.seed(random_state)

        # Diverse parameters per client
        fedxgbllr__xgb_params = dict(
            objective="reg:squarederror",
            eval_metric=["rmse"],
            n_estimators=args.n_estimators,    # WARNING: The final assembly will consist of ```clients``` * ```n_estimators``` - be careful on memory consumption
            learning_rate=np.exp(np.random.uniform(np.log(1e-2), np.log(0.3))),  # Log-uniform distribution
            max_depth=np.random.randint(3, 10),
            subsample=np.random.uniform(0.6, 1.0),
            colsample_bytree=np.random.uniform(0.6, 1.0),
            min_child_weight = np.random.randint(1, 10),
            reg_lambda=np.exp(np.random.uniform(np.log(1e-3), np.log(10.0))),  # Log-uniform distribution,
            reg_alpha=np.exp(np.random.uniform(np.log(1e-3), np.log(10.0))),  # Log-uniform distribution
            random_state=random_state, 
            early_stopping_rounds=None, 
            enable_categorical=False,
            nthread=1, # Ensure XGBoost uses a single thread
        )

        fedxgbllr__cnn_params = dict(
            trees_per_client=args.n_estimators
        )

        evaluate_fun_params = dict(
            unit='#e-Bikes (Inbound / Outbound)',
            metrics_funs=[smape, wape],
        )

        client = FedXGBllr_Client(
            federation_name=args.federation,
            oid=str(client_id),
            device=torch.device('cpu'),
            dataset_params=dataset_params,
            xgb_params=fedxgbllr__xgb_params,
            cnn_params=fedxgbllr__cnn_params,
            evaluate_fun=evaluate_model_multihead,
            evaluate_fun_params=evaluate_fun_params,
        )
    
        logger.debug(f"[client_fn] Created client {client_id}")
        return client


    # Generate model names/directories
    global_model_name = f'fedxgb_{args.federation}.flwr_global.epoch{{0}}.pth'
    model_save_path = (
        CFG_DATA_DIR/
        'pth'/
        f'fedxgb_ver{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{args.federation}_fraction_fit={args.fraction_fit}_fraction_eval={args.fraction_eval}_proximal_mu={args.mu}'

    )
    model_save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f'[server] Saving model to: {model_save_path}')

    data_save_path = (
        CFG_DATA_DIR/
        'pth'/
        f'fedxgb_{args.federation}'
    )
    data_save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f'[server] Saving datasets to: {data_save_path}')

    log_save_path = (
        CFG_LOGFILE_DIR / 
        f'{args.federation}_flwr_simulation'
    )
    log_save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f'[server] Saving logs to: {log_save_path}')

    # Define strategy
    strategy_params = dict(
        save_path = model_save_path,
        model_name = global_model_name,
        load_check = False,
        early_stop = args.early_stop,
        patience = args.patience,
        # 
        model_parameters = None,
        initial_parameters = None,
        # 
        proximal_mu = args.mu,
        local_epochs = args.local_epochs,
        fraction_fit = args.fraction_fit,
        # 
        fraction_evaluate = args.fraction_eval,
        min_available_clients = len(client_ids),
        min_fit_clients = max(2, int(len(client_ids) * args.fraction_fit)),
        min_evaluate_clients = max(2, int(len(client_ids) * args.fraction_eval)),
    )

    fedxgbllr__cnn_params = dict(
        num_clients=len(client_ids),
        in_channels=2,      # The prediction outcomes are treated as a single sequence of values
        conv_channels=args.conv_channels,   
        out_channels=2,     # Represents a single final prediction value, suitable for regression
        fc_layers=args.fc_layers,
        dropout_rate=args.dropout_rate
    )
    
    # Run simulation
    # This runs everything in-process with minimal threading
    logger.info("Starting simulation...\n")
    logger.info('[server] Using ```FedXGBllr``` aggregation strategy...')
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_ids),
        config=fl.server.ServerConfig(
            num_rounds=args.num_rounds
        ),
        strategy=FedXGBllr(
            cnn_params=fedxgbllr__cnn_params,
            accept_failures=False,
            **strategy_params
        ),
        client_resources={
            "num_cpus": 4,  # Limit resources per client
            "num_gpus": 0,
        },
    )

    # Print summary
    if history.losses_distributed:
        logger.info(f"Final loss: {history.losses_distributed[-1][1]:.4f}")
    
    logger.info("\nLoss history:")
    for round_num, loss in history.losses_distributed:
        logger.info(f"Round {round_num}: {loss:.4f}")
