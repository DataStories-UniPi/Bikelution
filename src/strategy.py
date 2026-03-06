from loguru import logger
from typing import (
    Any, 
    Dict, 
    List, 
    Optional, 
    Tuple, 
    Union
)

import torch
import numpy as np
import xgboost as xgb
import flwr as fl
from flwr.common import (
    FitRes, 
    Parameters, 
    Scalar, 
    EvaluateRes
)
from flwr.server.strategy import FedProx
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

import helper as hl


class CrossSiloFedProx(FedProx):
    def __init__(self, save_path, model_name, load_check, early_stop, patience, model_parameters, local_epochs, ndigits=10, **kwargs):
        self.ndigits = ndigits
        self.save_path = save_path
        self.model_name = model_name
        self.early_stop = early_stop
        self.patience = patience
        #
        (
            self.global_inbound_norm_const,
            self.global_outbound_norm_const,
        ) = (
            np.nan,
            np.nan
        )
        
        self.train_loss_aggregated = []
        self.dev_loss_aggregated = []
        # 
        self.latest_round = 0
        self.local_epochs = local_epochs
        
        if load_check:
            self.latest_round = model_parameters['round'] + 1
            self.train_loss_aggregated = model_parameters['loss']
            self.dev_loss_aggregated = model_parameters['dev_loss']
            self.global_inbound_norm_const = model_parameters['global_inbound_norm_const']
            self.global_outbound_norm_const = model_parameters['global_outbound_norm_const']

        super().__init__(**{
            **kwargs,
            'on_fit_config_fn': self.fit_config_fn,
            'on_evaluate_config_fn': self.evaluate_config_fn
        })

    def fit_config_fn(self, rnd: int):
        return {
            'round': rnd,
            'save_path': str(self.save_path),
            'local_epochs': self.local_epochs,
            'early_stop': self.early_stop,
            'patience': self.patience,
            'global_inbound_norm_const':self.global_inbound_norm_const,
            'global_downtime_norm_const':self.global_outbound_norm_const,

        }
    
    def evaluate_config_fn(self, rnd: int):
        return {
            'round': rnd,
            'save_path': str(self.save_path),
            'global_inbound_norm_const':self.global_inbound_norm_const,
            'global_outbound_norm_const':self.global_outbound_norm_const,
        }

    def save_model(self, rnd, aggregated_weights):
        model_dict = dict({
            'parameters':fl.common.parameters_to_ndarrays(aggregated_weights[0]),
            'round': rnd,
            'loss': self.train_loss_aggregated,
            'dev_loss': self.dev_loss_aggregated,
            'global_inbound_norm_const':self.global_inbound_norm_const,
            'global_outbound_norm_const':self.global_outbound_norm_const,
        })
        torch.save(model_dict, self.save_path/self.model_name.format(rnd))

        return aggregated_weights

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        curr_round = self.latest_round + server_round
        
        # Filter results for clients that actually contributed to the federation
        valid_results = [result for result in results if result[1].num_examples > 0]

        # If no clients contribute to the federation, maintain current (global) model parameters
        if not valid_results:
            logger.warning(f'[strategy][aggregate_fit] Round {curr_round} - No clients contributed in this round.')
            return None, {}
        
        # Call aggregate_fit from base class (FedProx)
        aggregated_weights = super().aggregate_fit(server_round, valid_results, failures)

        # Update aggregated metrics (train/dev loss)
        self.train_loss_aggregated.append(hl.weighted_sum(valid_results, 'train_loss'))
        self.dev_loss_aggregated.append(hl.weighted_sum(valid_results, 'dev_loss'))

        # Update data statistics (global scaling/normalization constants)
        self.global_inbound_norm_const = hl.weighted_sum(valid_results, 'local_inbound_norm_const')
        self.global_outbound_norm_const = hl.weighted_sum(valid_results, 'local_outbound_norm_const')
        logger.info(f"[strategy][aggregate_fit] Round {curr_round} - global_inbound_norm_const = {self.global_inbound_norm_const}")
        logger.info(f"[strategy][aggregate_fit] Round {curr_round} - global_outbound_norm_const = {self.global_outbound_norm_const}")

        # Save aggregated_weights
        logger.info(f'[strategy][aggregate_fit] Round {curr_round} - Saving aggregated_weights...')
        self.save_model(curr_round, aggregated_weights)

        return aggregated_weights
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        curr_round = self.latest_round + server_round
        
        # Filter results for clients that actually contributed to the federation
        valid_results = [result for result in results if result[1].num_examples > 0]

        # If no clients contribute to the federation, maintain current (global) model parameters
        if not valid_results:
            logger.warning(f'[strategy][aggregate_evaluate] Round {curr_round} - No clients contributed in this round.')
            return None, {}

        # Call aggregate_evaluate from base class (FedProx)
        loss_aggregated, metrics = super().aggregate_evaluate(curr_round, valid_results, failures)

        # Weigh (global/binned) accuracy of each client by number of examples used
        accuracy_aggregated_all = hl.weighted_sum(valid_results, 'test_acc')
        accuracy_aggregated_binned = hl.weighted_sum(valid_results, 'test_acc_binned')
        logger.info(f'[strategy][aggregate_evaluate] Round {curr_round} - {loss_aggregated = }; {accuracy_aggregated_all = }; {accuracy_aggregated_binned = }')

        # Output aggregated metrics
        return round(loss_aggregated, ndigits=self.ndigits), {
            **metrics, 
            'accuracy': np.around(accuracy_aggregated_all, decimals=self.ndigits),
            'accuracy_binned': np.around(accuracy_aggregated_binned, decimals=self.ndigits),
        }


class FedXGBllr(CrossSiloFedProx):
    """Configurable FedXGBllr strategy implementation."""

    def __init__(self, cnn_params, save_path, model_name, load_check, early_stop, patience, model_parameters, local_epochs, ndigits=10, **kwargs) -> None:
        """Federated XGBoost [Ma et al., 2023] strategy.

        Implementation based on https://arxiv.org/abs/2304.07537.
        Forked from https://github.com/adap/flower/blob/main/baselines/hfedxgboost/hfedxgboost/strategy.py
        """
        # Total number of clients that will ever be registered
        self._total_clients = kwargs.get("min_available_clients", 2)

        # FedXGBllr-specific parameters
        self.fedxgbllr__cnn_params = cnn_params

        self._fraction_fit_xgb = 1.0
        self._fraction_eval_xgb = 1.0
        self._fraction_fit_cnn = kwargs.get("fraction_fit", 1.0)
        self._fraction_eval_cnn = kwargs.get("fraction_evaluate", 1.0)
        
        kwargs['fraction_fit'] = self._fraction_fit_xgb
        logger.debug(f'Initial fraction fit (XGBoost) = {self._fraction_fit_xgb}')
        logger.debug(f'Defined fraction fit (1D-CNN) = {self._fraction_fit_cnn}')

        super().__init__(
            save_path, 
            model_name, 
            load_check, 
            early_stop, 
            patience, 
            model_parameters, 
            local_epochs,
            ndigits, 
            **kwargs
        )        

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f'FedXGBllr(accept_failures={self.accept_failures})'
        return rep

    def fit_config_fn(self, rnd: int):
        fitins = super().fit_config_fn(rnd)

        # In first FL round, send the instructions of FedXGBllr along with the configuration of the 1D-CNN
        logger.debug(f"Round {rnd} - Sending fit settings to current participants...")
        return {
            **fitins,
            **self.fedxgbllr__cnn_params
        }

    def evaluate_config_fn(self, rnd: int):
        evaluateins = super().evaluate_config_fn(rnd)

        logger.debug(f"Round {rnd} - Sending evaluation settings to current participants...")
        return {
            **evaluateins,
            **self.fedxgbllr__cnn_params
        }
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[str, Parameters]]:
        # Set fraction_fit based on FL round
        ## If its the first round of the federation, set fraction_fit to 1.0 (fully synchronous FL)
        if server_round == 1:
            self.fraction_fit = self._fraction_fit_xgb
        ## Otherwise, set fraction_fit to user defined value (asynchronous FL)
        else:
            self.fraction_fit = self._fraction_fit_cnn
        
        # Adjust client pool accordingly
        logger.debug(f"Using fraction_fit = {self.fraction_fit}")
        self.min_fit_clients = max(2, int(self._total_clients * self.fraction_fit))

        # Call parent method with updated fraction_fit / min_fit_clients values
        return super().configure_fit(server_round, parameters, client_manager)
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[str, Parameters]]:
        # Set fraction_evaluate based on FL round
        ## If its the first round of the federation, set fraction_evaluate to 1.0 (fully synchronous FL)
        if server_round == 1:
            self.fraction_evaluate = self._fraction_eval_xgb
        ## Otherwise, set fraction_evaluate to user defined value (asynchronous FL)
        else:
            self.fraction_evaluate = self._fraction_eval_cnn
        
        # Adjust client pool accordingly
        logger.debug(f"Using fraction_evaluate = {self.fraction_evaluate}")
        self.min_evaluate_clients = max(2, int(self._total_clients * self.fraction_evaluate))

        # Call parent method with updated fraction_evaluate / min_fit_clients values
        return super().configure_evaluate(server_round, parameters, client_manager)

    def save_xgb_models(self, aggregated_trees) -> None:
        reconstructed_trees = []
        
        for tree_bytes, oid in zip(aggregated_trees[::2], aggregated_trees[1::2]):
            tree = xgb.XGBRegressor()
            tree.load_model(bytearray(tree_bytes))
    
            reconstructed_trees.append(
                (
                    tree, 
                    oid.decode('utf-8')
                )
            )
        
        np.save(
            file=self.save_path/self.model_name[:-4].format('0.xgb_trees'),
            arr=np.array(
                reconstructed_trees,
                dtype='object'
            )
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        curr_round = self.latest_round + server_round

        # Filter results for clients that actually contributed to the federation
        valid_results = [result for result in results if result[1].num_examples > 0]

        # If no clients contribute to the federation, maintain current (global) model parameters
        if not valid_results:
            logger.warning(f'[strategy][aggregate_fit] Round {curr_round} - No clients contributed in this round.')
            return None, {}

        # If its the first FL round, aggregate the XGBoost trees (i.e., combine them into a list of <XGBRegressor, int>)
        if server_round == 1:
            # Create empty list; to be filled with Tuple[XGBRegressor, int]
            aggregated_trees = []

            # Sort XGBoost trees, by client identifers
            valid_results = sorted(valid_results, key=lambda l: l[1].parameters.tensors[1].decode('utf-8'))

            for _, fit_res in valid_results:
                aggregated_trees.extend(fit_res.parameters.tensors)

            # Update aggregated metrics (train/dev loss)
            self.train_loss_aggregated.append(hl.weighted_sum(valid_results, 'train_loss'))
            self.dev_loss_aggregated.append(hl.weighted_sum(valid_results, 'dev_loss'))

            # Save aggregated trees
            logger.info(f'[strategy][aggregate_fit] Round {curr_round} - Saving aggregated trees...')
            self.save_xgb_models(aggregated_trees)

            # Aggregate custom metrics if aggregation fn was provided
            metrics_aggregated = {}
            if self.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            elif server_round == 1:  # Only log this warning once
                logger.warning("No fit_metrics_aggregation_fn provided")

            return Parameters(tensors=aggregated_trees, tensor_type='bytearray'), metrics_aggregated

        # Call aggregate_fit from base class (FedProx)
        aggregated_weights = super().aggregate_fit(server_round, valid_results, failures)

        # Save aggregated_weights
        logger.info(f'[strategy][aggregate_fit] Round {curr_round} - Saving aggregated_weights...')
        self.save_model(curr_round, aggregated_weights)

        return aggregated_weights
