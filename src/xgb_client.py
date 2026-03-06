import json
import pathlib
from loguru import logger
from typing import (
    List,
    Tuple
)

import numpy as np 
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error

import torch
from torch.nn import MSELoss
from torch.optim import Adam

import flwr as fl
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersRes,
    GetParametersIns,
    Status,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

import helper as hl
import models as ml
import dataset as ds
import train as tr


class FedXGBllr_Client(fl.client.Client):
    def __init__(self, federation_name, oid, device, dataset_params, xgb_params, cnn_params, evaluate_fun, evaluate_fun_params) -> None:
        # Related to data fetching / processing
        (
            self.federation_name,
            self.dataset_params,
            self.oid
        ) = (
           federation_name,
           dataset_params,
           oid
        )
        
        # Set random seed for PyTorch
        # NOTE: Ensure diverse model initialization, as well as reproducible results...
        torch.manual_seed(xgb_params['random_state'])

        # Related to model instantiation / training        
        (
            self.device, 
            self.criterion,
            self.evaluate_fun, 
            self.evaluate_fun_params,
            self.xgb_params, 
            self.cnn_params
        ) = (
            device, 
            MSELoss(),
            evaluate_fun, 
            evaluate_fun_params,
            xgb_params, 
            cnn_params
        )

        # File names for XGB / 1D-CNN models and datasets..
        self.xgb_model_name = f'fed{"xgb"}_{self.oid}.flwr_local.round0.json'
        self.xgb_array_name = f'fed{"xgb"}_{self.federation_name}.flwr_global.epoch0.xgb_trees.npy'
        self.cnn_model_name = f'fed{"xgb"}_{self.federation_name}_{self.oid}.flwr_local.round{{0}}.epoch{{1}}.pth'

        # Define placeholders XGB / 1D-CNN models
        # NOTE: These will be dynamically allocated during runtime
        (
            self.xgb_model,
            self.cnn_model
        ) = (
            None,
            None
        )

        logger.info(f'Client #{self.oid} - Initialized client instance.')

    def get_parameters__xgb_model(self) -> Parameters:
        return Parameters(
            tensors=[
                bytes(
                    self.xgb_model.get_booster().save_raw()
                ), 
                self.oid.zfill(15).encode('utf-8')
            ], 
            tensor_type='bytearray'
        ) 
    
    def get_parameters__cnn_model(self) -> Parameters:
        return ndarrays_to_parameters(
            hl.get_parameters(
                self.cnn_model
            )
        )

    def get_parameters(self, ins:GetParametersIns) -> GetParametersRes:
        # If it's the first FL round, ```send``` the ```tree```, along with the ```identifier``` of the client...
        if self.cnn_model is None:
            logger.warning(f'Client #{self.oid} - The FedXGBllr 1D-CNN model is not initialized!')
            return GetParametersRes(
                status=Status(Code.OK, ''),
                parameters=Parameters(
                    tensors=[], 
                    tensor_type='numpy.ndarray'
                ) 
            )
        
        # ...otherwise, ```send``` the parameters of the ```1D CNN```
        logger.info(f'Client #{self.oid}] - Getting parameters of local FedXGBllr 1D-CNN...')
        return GetParametersRes(
            status=Status(Code.OK, ''),
            parameters=self.get_parameters__cnn_model()
        )

    def set_parameters__xgb(self, parameters) -> List[Tuple[xgb.XGBRegressor, int]]:
        xgb_models = []
        logger.info(f'Client #{self.oid} - Reconstructing aggregated XGBoost bytearray...')

        aggregated_trees = parameters.tensors
        for tree_bytes, oid in zip(aggregated_trees[::2], aggregated_trees[1::2]):
            tree = xgb.XGBRegressor()
            tree.load_model(bytearray(tree_bytes))
            xgb_models.append(
                (
                    tree, 
                    oid.decode('utf-8')
                )
            )
        
        # Sanity Check - Ensure that XGBoost trees are sorted w.r.t. client ID
        xgb_models_ids = [tree_id for _, tree_id in xgb_models]
        is_sorted_xgb_models_ids = np.all(
            xgb_models_ids[:-1] <= xgb_models_ids[1:]
        )

        logger.info(f'Client #{self.oid} - Received {len(xgb_models)} trees; Sorted (w.r.t. ID): {is_sorted_xgb_models_ids}')
        assert is_sorted_xgb_models_ids

        # Return reconstructed XGBoost trees
        return xgb_models

    def set_parameters__cnn(self, parameters) -> str:
        # Load parameters
        parameters_ndarrays = parameters_to_ndarrays(parameters)

        logger.info(f'Client #{self.oid} - Setting global FedXGBllr 1D-CNN parameters...')
        return hl.set_parameters(self.cnn_model, parameters_ndarrays)

    def prox_loss(self, local_model, xb, yb, criterion, *args, **kwargs) -> Tuple[float, float]:
        y_pred = local_model(xb, *args).float()
        
        # proximal_term = 0.0
        proximal_term = torch.tensor(0.0, device=self.device).float()
        
        # Compute the proximal term as the squared L2 norm of weight differences
        for local_weights, global_weights in zip(local_model.parameters(), kwargs['global_model_parameters']):
            proximal_term += (local_weights - global_weights).norm(2).pow(2)
        
        # Calculate total loss with the proximal term scaled by proximal_mu
        loss = criterion(y_pred, yb) + (kwargs['proximal_mu'] / 2) * proximal_term
        return y_pred, loss
    
    def fit(self, ins:FitIns) -> FitRes:
        # Unpack FitIns (model parameters, configution values)
        parameters, config = ins.parameters, ins.config
        logger.debug(f'Client #{self.oid} - {config=}')

        # Load the training / validation / test datasets (tabular format - for XGBoost training)
        (
            xgb_train_dataset, 
            xgb_dev_dataset, 
            xgb_test_dataset
        ) = hl.ebike_load_data_xgboost(
            **{
                'dataset_name': self.federation_name,
                'dataset_params': self.dataset_params,
                'oid': self.oid,
            }
        )
        logger.info(f'Client #{self.oid} - Loaded eBike train/dev/test dataset (tabular format)...')
   

        # If it's the first FL round...
        if config['round'] == 1:
            logger.warning(f'Client #{self.oid} - This is the first round of the federation.')
            
            # Train XGBoost model
            self.xgb_model = xgb.XGBRegressor(
                **self.xgb_params,
            )

            logger.info(f'Client #{self.oid} - Training XGBoost model...')
            self.xgb_model.fit(
                **xgb_train_dataset,
                eval_set=[
                    xgb_train_dataset.values(),
                    xgb_dev_dataset.values(),
                ], 
                verbose=False
            )

            self.xgb_model_score = self.xgb_model.score(**xgb_test_dataset)
            logger.info(f'Client #{self.oid} - XGBoost model R^2 score = {self.xgb_model_score:.3f}')
                        
            self.xgb_model.save_model(
                pathlib.Path(config['save_path']) / self.xgb_model_name
            )  # Saves as JSON
            logger.info(f'Client #{self.oid} - Saving local XGBoost model...')
                      
            # Send XGBoost parameters
            return FitRes(
                status=Status(
                    Code.OK, 
                    f'Client #{self.oid} - Sending XGBoost model to aggregation server...'
                ), 
                parameters=self.get_parameters__xgb_model(), 
                num_examples=len(xgb_train_dataset['X']), 
                metrics={
                    'train_loss':float(self.xgb_model.evals_result_['validation_0']['rmse'][-1]),
                    'dev_loss':float(self.xgb_model.evals_result_['validation_1']['rmse'][-1])
                }
            )
        

        # Load aggregated XGBoost tree array
        self.xgb_trees = np.load(
            pathlib.Path(config['save_path']) / self.xgb_array_name,
            allow_pickle=True
        )

        # Use the aggregated trees and training / validation datasets of the XGBoost model 
        # to create the training / validation datasets of the 1D-CNN model
        dataloader_save_params = {
            'save_path':pathlib.Path(config['save_path']),
            'dataset_name':f'fedxgb_{self.federation_name}',
            'in_channels':config['in_channels'],
            'trees_per_client':self.cnn_params['trees_per_client'],
            'parquet_bs':self.dataset_params['parquet_bs'],
            'bs':self.dataset_params['bs'],
            'oid':self.oid,
        }

        logger.info(f'Client #{self.oid} - Round {config["round"]} - Creating 1D-CNN train dataset...')
        cnn_train_dataset_params = hl.fedxgbllr_cnn_create_tensor_dataset(
            self.xgb_trees,
            xgb_train_dataset,
            {
                **dataloader_save_params,
                'dataset_slice':'train',
            },
            **{
                'enable_categorical': self.xgb_params['enable_categorical']
            }
        )

        logger.info(f'Client #{self.oid} - Round {config["round"]} - Creating 1D-CNN dev dataset...')
        cnn_dev_dataset_params = hl.fedxgbllr_cnn_create_tensor_dataset(
            self.xgb_trees,
            xgb_dev_dataset,
            {
                **dataloader_save_params,
                'dataset_slice':'dev',
            },
            **{
                'enable_categorical': self.xgb_params['enable_categorical']
            }
        )

        # Drop the reference to the clients' test dataset - the CNN train/dev dataset is created
        del xgb_test_dataset, xgb_dev_dataset, xgb_train_dataset

        # Initialize 1D-CNN model...
        # Download FedXGBllr 1D-CNN configuration from the server
        self.cnn_params.update({
            'num_clients':config['num_clients'],
            'in_channels':config['in_channels'],
            'conv_channels':config['conv_channels'],
            'out_channels':config['out_channels'],
            'dropout_rate':config['dropout_rate'],
            'fc_layers':[int(i) for i in config['fc_layers'].split(',') if i],
        })

        self.cnn_model = ml.FedXGBllrCNN(**self.cnn_params)
        logger.info(f'Client #{self.oid} - Created FedXGBllrCNN instance! \n{self.cnn_model=}')
        

        if config['round'] > 2:
            # Initialize local model with the parameters of the global model
            self.set_parameters__cnn(parameters)
        
        model_save_path = (
            pathlib.Path(config['save_path']) / 
            self.cnn_model_name.format(config['round'], '{0}')
        )
        logger.info(f'Client #{self.oid} - Saving model to: {model_save_path}')

        # Perform training for K epochs
        global_model_parameters = [parameter.detach().clone() for parameter in self.cnn_model.parameters()]
        
        optimizer_fun = Adam(self.cnn_model.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=1e-3)   # Adapted from the FedXGBllr paper
        criterion_fun, criterion_fun_params, save_current_params, early_stop_params = self.prox_loss, {
            'global_model_parameters':global_model_parameters,
            'proximal_mu':config['proximal_mu']
        }, {
            'path':str(model_save_path),
            'round':config['round'],
        }, {
            'patience':config['patience'],
            'save_best':False,
        }

        # Create DataLoaders for training 1D-CNN model
        cnn_train_dataset = ds.FedXGBllrDataset(**cnn_train_dataset_params)
        cnn_dev_dataset = ds.FedXGBllrDataset(**cnn_dev_dataset_params)

        # Proceed with FL training of FedXGBllr 1D-CNN model, as usual...
        logger.debug(f'Client #{self.oid} - Memory report (train start) - {hl.mem_report()}')
        train_losses, dev_losses = tr.train_model(
            self.cnn_model, self.device, 
            self.criterion, optimizer_fun, config['local_epochs'],
            cnn_train_dataset, cnn_dev_dataset, 
            criterion_fun=criterion_fun, criterion_fun_params=criterion_fun_params, 
            evaluate_cycle=-1, early_stop=config['early_stop'], save_current=True,
            evaluate_fun=self.evaluate_fun, evaluate_fun_params=self.evaluate_fun_params, 
            save_current_params=save_current_params, early_stop_params=early_stop_params,
        )
    
        fit_tldr = dict(
            train_loss=float(train_losses[-1]),
            **(
                {'dev_loss':float(dev_losses[-1])} if dev_losses else {}
            ),
        )
        logger.debug(f'Memory report (train end) - {hl.mem_report()}')
        logger.info(f'Client #{self.oid} - Round {config["round"]} - {train_losses = }')
        logger.info(f'Client #{self.oid} - Round {config["round"]} - {dev_losses = }')

        # Send updated model/metrics to the aggregation server
        return FitRes(
            status=Status(Code.OK, f''), 
            parameters=self.get_parameters__cnn_model(), 
            num_examples=cnn_train_dataset._total_samples, 
            metrics=fit_tldr
        )
    
    def evaluate(self, ins:EvaluateIns) -> EvaluateRes:
        # Unpack EvaluateIns (model parameters, configution values)
        parameters, config = ins.parameters, ins.config
        logger.info(f'Client #{self.oid} - {config=}')

        # Load the test dataset (tabular format - for XGBoost training)
        (
            _, 
            _, 
            xgb_test_dataset
        ) = hl.ebike_load_data_xgboost(
            **{
                'dataset_name': self.federation_name,
                'dataset_params': self.dataset_params,
                'oid': self.oid,
                'dataset_slice':[
                    'test',
                ]
            }
        )


        # If it's the first FL round...
        if config['round'] == 1:
            # Receive XGBoost models 
            logger.info(f'Client #{self.oid} - Round {config["round"]} - Receiving aggregted XGBoost trees...')
            
            # Save the reconstructed XGBoost assemble, as received from the server.
            # NOTE: Trained XGBoost instances are immutable, therefore they can be reconstructed only when needed...
            # NOTE: Given that it is already saved server-side and targets the same directory, it is redundant and
            # ...required only during deployment/production, where each client has its own (distinct) directory tree.
            np.save(
                pathlib.Path(config['save_path']) / self.xgb_array_name,
                self.set_parameters__xgb(parameters)
            )

            # Load XGBoost model
            self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
            self.xgb_model.load_model(
                pathlib.Path(config['save_path']) / self.xgb_model_name
            )

            logger.info(f'Client #{self.oid} - Round {config["round"]} - Evaluating local XGBoost model...')
            xgb_model_eval_loss, xgb_model_eval_metrics = float(
                self.criterion(
                    y_hat := torch.from_numpy(
                        self.xgb_model.predict(xgb_test_dataset['X'])
                    ), 
                    y := torch.from_numpy(
                        xgb_test_dataset['y'].values
                    )
                )
            ), dict(
                test_acc=float(root_mean_squared_error(y, y_hat)),
                test_acc_binned=json.dumps(
                    np.array(
                        [
                            metrics_fun(y.numpy(), y_hat.numpy()) for metrics_fun in self.evaluate_fun_params['metrics_funs']
                        ]
                    ).astype(float).tolist()
                )
            )

            # Return metrics of XGBoost model(s)
            logger.info(f'Client #{self.oid} - Sending XGBoost evaluation results to aggregation server...')
            return EvaluateRes(
                status=Status(Code.OK, ''),
                loss=xgb_model_eval_loss,
                num_examples=len(xgb_test_dataset['X']),
                metrics=xgb_model_eval_metrics,
            )
        

        # Load aggregated XGBoost tree array
        self.xgb_trees = np.load(
            pathlib.Path(config['save_path']) / self.xgb_array_name,
            allow_pickle=True
        )

        # Initialize 1D-CNN model...
        # Download FedXGBllr 1D-CNN configuration from the server
        self.cnn_params.update({
            'num_clients':config['num_clients'],
            'in_channels':config['in_channels'],
            'conv_channels':config['conv_channels'],
            'out_channels':config['out_channels'],
            'dropout_rate':config['dropout_rate'],
            'fc_layers':[int(i) for i in config['fc_layers'].split(',') if i],
        })

        self.cnn_model = ml.FedXGBllrCNN(**self.cnn_params)
        logger.info(f'Client #{self.oid} - Created FedXGBllrCNN instance! \n{self.cnn_model=}')
        
        # Download the parameters of the global FedXGBllr 1D-CNN model
        self.set_parameters__cnn(parameters)

        # Use the aggregated trees and testing datasets of the XGBoost model 
        # to create the testing datasets of the 1D-CNN model
        dataloader_save_params = {
            'save_path':pathlib.Path(config['save_path']),
            'dataset_name':f'fedxgb_{self.federation_name}',
            'in_channels':config['in_channels'],
            'trees_per_client':self.cnn_params['trees_per_client'],
            'parquet_bs':self.dataset_params['parquet_bs'],
            'bs':self.dataset_params['bs'],
            'oid':self.oid,
        }

        logger.info(f'Client #{self.oid} - Round {config["round"]} - Creating 1D-CNN test dataset...')
        self.cnn_test_dataset_params = hl.fedxgbllr_cnn_create_tensor_dataset(
            self.xgb_trees,
            xgb_test_dataset,
            {
                **dataloader_save_params,
                'dataset_slice':'test',
            },
            **{
                'enable_categorical': self.xgb_params['enable_categorical']
            }
        )

        # Drop the reference to the clients' test dataset
        del xgb_test_dataset

        # Create DataLoaders for training 1D-CNN model
        cnn_test_dataset = ds.FedXGBllrDataset(**self.cnn_test_dataset_params)

        # Evaluate the global model on the test set of the current FL round -- NEEDS REVISION (multiple steps, not just an average of each one)
        logger.debug(f'Client #{self.oid} - Memory report (eval start) - {hl.mem_report()}')
        test_loss, test_avg_err_all, test_avg_err_binned = self.evaluate_fun(
            self.cnn_model, self.device, self.criterion, cnn_test_dataset,
            desc=f'Client #{self.oid} - Round #{config["round"]} - ADE @ Test Set...', **self.evaluate_fun_params
        )
        eval_tldr = dict(
            test_acc=float(test_avg_err_all),
            test_acc_binned=json.dumps(test_avg_err_binned.astype(float).tolist())
        )
        logger.debug(f'Client #{self.oid} - Memory report (eval end) - {hl.mem_report()}')
        logger.info(f'Client #{self.oid} - Round {config["round"]} - {test_loss = }; {test_avg_err_all = }; {test_avg_err_binned = }')

        # Send updated eval metrics to the aggregation server
        return EvaluateRes(
            status=Status(Code.OK, f''), 
            loss=float(test_loss),
            num_examples=cnn_test_dataset._total_samples, 
            metrics=eval_tldr
        )
