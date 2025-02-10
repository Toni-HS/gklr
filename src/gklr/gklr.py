"""GKLR main module."""
from __future__ import annotations
from typing import Optional, Any, Dict, List

import sys
import gc
import time

import numpy as np
from pympler import asizeof
import pandas as pd

from .logger import *
from .config import Config
from .kernel_utils import *
from .kernel_estimator import KernelEstimator
from .kernel_calcs import KernelCalcs
from .nested_kernel_calcs import NestedKernelCalcs
from .nested_kernel_estimatory import NestedKernelEstimator
from .kernel_matrix import KernelMatrix

from collections import Counter # To count occurrences of numbers in the vector of vectors
from typing import Set, Tuple # To return multiple values

valid_gklr_params = ["n_jobs", "nystrom", "compression", "ridge_leverage_lambda", "nystrom_sampling"]

class KernelModel:
    """Main class for GKLR models."""

    def __init__(self, model_params: Optional[Dict[str, Any]] = None) -> None:
        """Constructor.

        Args:
            model_params: A dict where the keys are the parameters of the kernel model and the value they contain.
                Default: None.
        """
        self._X = None
        self.choice_column = None
        self.attributes = None
        self._Z = None
        self._K = None
        self._K_test = None
        self._alpha = None
        self.alpha_shape = None
        self.lambd = None
        self.lambd_shape = None
        self.n_parameters = 0
        self.results = None
        self.nests = None
        self.nested = False

        if model_params == None:
            self._model_params = None
        else:
            # TODO: Check parameters

            # TODO: Check if the nests exist and in that case, check if they are correctly defined

            if "nests" in model_params:
                msg = "The nests will be ignored."
                nests = model_params["nests"]
                if not isinstance(nests, list):
                    msg += " The nests must be in a list."
                    logger_warning(msg)
                elif len(nests) == 0:
                    msg += " There must be nests."
                    logger_warning(msg)
                elif len(nests) == 1:
                    logger_warning(msg)
                else:
                    done = False
                    for nest in nests:
                        if not done:
                            if not isinstance(nest, list):
                                msg += " The nests must be lists."
                                logger_warning(msg)
                                done = True
                            elif len(nest) == 0:
                                msg += " The nests cannot be empty."
                                logger_warning(msg)
                                done = True
                            elif not all(isinstance(num, int) for num in nest):
                                logger_warning(msg)
                                done = True

                    if not done:
                        self.nests = nests
                        self.nested = True

            self._model_params = model_params

        self.config = Config()

        logger_debug("KernelModel initialized.")  

    def _set_kernel_params(self, hyperparams: Dict[str, Any]) -> None:
        """Set the kernel parameters.

        Store the hyperparameters of the GKLR model in a config object and
        a dict with the parameters to be passed to the kernel function.

        Args:
            hyperparams: A dict where the keys are the hyperparameters passed 
                to the GKLR object and their value.
        """
        kernel_params = hyperparams.copy()
        list_kernel_params = list(kernel_params.keys())

        if "kernel" in kernel_params and kernel_params["kernel"] in valid_kernel_list:
            self.config.set_hyperparameter("kernel", kernel_params.pop("kernel"))
            list_kernel_params.remove("kernel")

        for param in list_kernel_params:
            if param in valid_gklr_params:
                self.config.set_hyperparameter(param, kernel_params.pop(param))
            elif param in valid_kernel_params:
                # Valid parameter for the kernel function
                pass
            else:   
                raise ValueError(f"Parameter {param} is not a valid KernelModel",
                    "parameter.")

        # Store the kernel function parameters
        self.config.set_hyperparameter("kernel_params", kernel_params)

    def _create_kernel_matrix(self,
                              X: pd.DataFrame,
                              choice_column: str,
                              attributes: Dict[int, List[str]],
                              config: Config,
                              Z: Optional[pd.DataFrame] = None,
                              train: bool = True,
    ) -> bool:
        """Creates a KernelMatrix object.

        Creates the KernelMatrix object and store it in a private variable.

        Args:
            X: Train dataset stored in a pandas DataFrame. Shape: (n_samples, n_features)
            choice_column: Name of the column of DataFrame `X` that contains the ID of chosen alternative.
            attributes: A dict that contains the columns of DataFrame `X` that are considered for each alternative.
                This dict is indexed by the ID of the available alternatives in the dataset and the values are list
                containing the names of all the columns considered for that alternative. 
            config: A Config object that contains the hyperparameters of the GKLR model.
            Z: Test dataset stored in a pandas DataFrame. Shape: (n_samples, n_features).
                Default: None
            train: A boolean that indicates if the kernel matrix to be created is for train or test data.
                Default: True.

        Returns:
            A bolean that indicates if the kerne matrix was successfully created.
        """
        success = True # TODO: Check conditions before create kernel, if not satisfied, then success is False
        # TODO: ensure_columns_are_in_dataframe
        # TODO: ensure_valid_variables_passed_to_kernel_matrix

        config.check_values()

        if train:
            self._K = KernelMatrix(X, choice_column, attributes, config, Z)
            self.n_parameters = self._K.get_num_cols() * self._K.get_num_alternatives() # One alpha vector per alternative
            self.alpha_shape = (self._K.get_num_cols(), self._K.get_num_alternatives())
        else:
            self._K_test = KernelMatrix(X, choice_column, attributes, config, Z)
        return success

    def get_kernel(self, dataset: str = "train") -> KernelMatrix | None:
        """Returns the train and/or test KernelMatrix object.

        Args:
            dataset: The kernel matrix to be retrieved. It can take the values: "train", "test" or "both".
                Default: "train".

        Returns:
            The KernelMatrix object.
        """
        if dataset == "train":
            return self._K
        elif dataset == "test":
            return self._K_test
        else:
            msg = "Dataset must be a value in: ['train', 'test', 'both']"
            logger_error(msg)
            raise ValueError(msg)

    def clear_kernel(self, dataset: str = "train") -> None:
        """Clear the kernel matrices previously computed.

        Removes the train and test kernel matrices and frees the memory.

        Args:
            dataset: The kernel matrix to be deleted. It can take the values: "train", "test" or "both".
                Default: "train".
        """
        if dataset == "train":
            self._K = None
        elif dataset == "test":
            self._K_test = None
            self._Z = None
        elif dataset == "both":
            self._K = None
            self._K_test = None
            self._X = None
            self._Z = None
        else:
            msg = "Dataset must be a value in: ['train', 'test', 'both']"
            logger_error(msg)
            raise ValueError(msg)
        gc.collect()
        return None

    def validate_nests(self, nests: List[List[int]], alternatives: List[int]) -> None:
        """Validate the nests with the alternatives.
        
        Args:
            nests: Nests defined by the user.
            alternatives: Choice column alternatives of the dataset.
        """
        flattened_counts = Counter(num for sublist in nests for num in sublist)
        flattened_set = set(flattened_counts.keys())
        alternatives_set = set(alternatives)
        duplicates = {num for num, count in flattened_counts.items() if count > 1}
        missing = alternatives_set - flattened_set
        extra = flattened_set - alternatives_set

        if missing:
            msg = f"Missing alternatives in nests: {', '.join(map(str, missing))}"
            logger_error(msg)
            raise ValueError(msg)
        if extra:
            msg = f"Extra alternatives in nests: {', '.join(map(str, extra))}"
            logger_error(msg)
            raise ValueError(msg)
        if duplicates:
            msg = f"Duplicates alternatives in nests: {', '.join(map(str, duplicates))}"
            logger_error(msg)
            raise ValueError(msg)




    def set_kernel_train(self,
                         X: pd.DataFrame,
                         choice_column: str,
                         attributes: Dict[int, List[str]],
                         hyperparams: Dict[str, Any],
                         verbose: int = 1,
    ) -> None:
        """Computes the kernel matrix for the train dataset.

        Processes the train dataset and creates the corresponding kernel matrix. The kernel matrix is encapsulated and
        stored using the KernelMatrix class.

        Args:
            X: Train dataset stored in a pandas DataFrame. Shape: (n_samples, n_features)
            choice_column: Name of the column of DataFrame `X` that contains the ID of chosen alternative.
            attributes: A dict that contains the columns of DataFrame `X` that are considered for each alternative.
                This dict is indexed by the ID of the available alternatives in the dataset and the values are list
                containing the names of all the columns considered for that alternative. 
            hyperparams: A dict where the keys are the hyperparameters passed to the kernel function and the value they
                contain.
            verbose: Indicates the level of verbosity of the function. If 0, no output will be printed. If 1, basic
                information about the time spent and the size of the matrix will be displayed. Default: 1.
        """
        if self.nested == True:
            self.validate_nests(self.nests, X[choice_column].unique())

        self.clear_kernel(dataset="both")
        self._set_kernel_params(hyperparams)
        start_time = time.time()
        success = self._create_kernel_matrix(X, choice_column, attributes, self.config, train=True)
        elapsed_time_sec = time.time() - start_time

        if success == 0:
            self.clear_kernel(dataset="train")
            msg = "The kernel matrix for the train set have NOT been created."
            logger_error(msg)
            raise RuntimeError(msg)
        else:
            self._X = X
            self.choice_column = choice_column
            self.attributes = attributes

        elapsed_time_str = elapsed_time_to_str(elapsed_time_sec)
        K_size, K_size_u = convert_size_bytes_to_human_readable(asizeof.asizeof(self._K))

        if verbose >= 1:
            print(f"The kernel matrix for the train set have been correctly created in {elapsed_time_str}. "
                  f"Size of the matrix object: {K_size} {K_size_u}")
            sys.stdout.flush()
        logger_debug(f"Kernel matrix for train dataset estimated in {elapsed_time_str}. Size: {K_size} {K_size_u}")
        return None

    def set_kernel_test(self,
                        Z: pd.DataFrame,
                        choice_column: Optional[str] = None,
                        attributes: Optional[Dict[int, List[str]]] = None,
                        verbose: int = 1,
    ) -> None:
        """Computes the kernel matrix test dataset.

        Processes the test dataset and creates the corresponding kernel matrix. The kernel matrix is encapsulated and
        stored using the KernelMatrix class.

        Args:
            Z: Test dataset stored in a pandas DataFrame. Shape: (n_samples, n_features)
            choice_column: Name of the column of DataFrame `Z` that contains the ID of chosen alternative.
            attributes: A dict that contains the columns of DataFrame `Z` that are considered for each alternative.
                This dict is indexed by the ID of the available alternatives in the dataset and the values are list
                containing the names of all the columns considered for that alternative. 
            verbose: Indicates the level of verbosity of the function. If 0, no output will be printed. If 1, basic
                information about the time spent and the size of the matrix will be displayed. Default: 1.
        """
        if self.nested == True:
            self.validate_nests(self.nests, Z[choice_column].unique())

        if self._X is None or self._K is None or self.choice_column is None or self.attributes is None:
            msg = "First you must compute the kernel for the train dataset using set_kernel_train()."
            logger_error(msg)
            raise RuntimeError(msg)

        self.clear_kernel(dataset="test")

        # Set default values for the input parameters
        choice_column = self.choice_column if choice_column is None else choice_column
        attributes = self.attributes if attributes is None else attributes

        start_time = time.time()
        success = self._create_kernel_matrix(self._X, choice_column, attributes, self.config, 
                                             Z=Z, train=False)
        elapsed_time_sec = time.time() - start_time

        if success == 0:
            msg = "The kernel matrix for the test set have not been created."
            logger_error(msg)
            raise RuntimeError(msg)
        else:
            self._Z = Z

        elapsed_time_str = elapsed_time_to_str(elapsed_time_sec)
        K_size, K_size_u = convert_size_bytes_to_human_readable(asizeof.asizeof(self._K_test))
        if verbose >= 1:
            print(f"The kernel matrix for the test set have been correctly created in {elapsed_time_str}. "
                  f"Size of the matrix object: {K_size} {K_size_u}")
            sys.stdout.flush()
        logger_debug(f"Kernel matrix for test dataset estimated in {elapsed_time_str}. Size: {K_size} {K_size_u}")
        return None

    def fit(self,
            alpha_params: Optional[np.ndarray] = None,
            lambd_params: Optional[np.ndarray] = None,
            pmle: str = "Tikhonov",
            pmle_lambda: float = 0,
            method: str = "L-BFGS-B",
            options: Optional[Dict[str, Any]] = None,
            verbose: int = 1,
    ) -> None:
        """Fit the kernel model.

        Perform the estimation of the kernel model and store post-estimation results.

        Args:
            alpha_params: Initial value of the parameters to be optimized.
                Shape: (num_cols_kernel_matrix, n_features). Default: None
            lambd_params: Initial value of the nests parameters to be optimized.
                Shape: (lambd_shape,). Default: None
            pmle: Penalization method. Default: None.
            pmle_lambda: Parameter for the penalization method. Default: 0
            method: Optimization method. Default: "L-BFGS-B".
            options: Options for the optimization method. Default: None.
            verbose: Indicates the level of verbosity of the function. If 0, no output will be printed. If 1, basic
                information about the time spent and the Log-likelihood value will be displayed. Default: 1.
        """
        if self._K is None or self.alpha_shape is None:
            msg = "First you must compute the kernel for the train dataset using set_kernel_train()."
            logger_error(msg)
            raise RuntimeError(msg)

        if self.nested == True:
            self.lambd_shape = len(self.nests)
            # Create the NestedKernelCalcs instance
            calcs = NestedKernelCalcs(K=self._K, nests=self.nests)
            # Create the NestedKernelEstimator instance
            estimator = NestedKernelEstimator(calcs=calcs, pmle=pmle, pmle_lambda=pmle_lambda, method=method, verbose=verbose)
        else:
            # Create the Calcs instance
            calcs = KernelCalcs(K=self._K)
            # Create the estimator instance
            estimator = KernelEstimator(calcs=calcs, pmle=pmle, pmle_lambda=pmle_lambda, method=method, verbose=verbose)

        if alpha_params is None:
            alpha_params = np.zeros(self.alpha_shape, dtype=DEFAULT_DTYPE)
        else:
            pass # TODO: check that there are self.n_parameters and then make a cast to self.alpha_shape

        lambd_at_1 = None
        if lambd_params is None and self.nested == True:
            lambd_params = np.ones(self.lambd_shape, dtype=DEFAULT_DTYPE)
            # Log-likelihood at one
            lambd_at_1 = np.ones(self.lambd_shape, dtype=DEFAULT_DTYPE)
        else:
            pass

        # Log-likelihood at zero
        alpha_at_0 = np.zeros(self.alpha_shape, dtype=DEFAULT_DTYPE)
        log_likelihood_at_zero = calcs.log_likelihood(alpha_at_0, lambd = lambd_at_1)

        # Initial log-likelihood
        initial_log_likelihood = calcs.log_likelihood(alpha_params, lambd = lambd_params)

        if verbose >= 1:
            print("The estimation is going to start...\n"
                  "Log-likelihood at zero: {ll_zero:,.4f}\n"
                  "Initial log-likelihood: {i_ll:,.4f}".format(ll_zero=log_likelihood_at_zero, i_ll=initial_log_likelihood))
            sys.stdout.flush()
        if verbose >= 2:
            print("Number of parameters to be estimated: {n_parameters:,d}".format(n_parameters=self.n_parameters))
            sys.stdout.flush()

        # Perform the estimation
        start_time = time.time()
        ## TODO: Concatenate alpha_params and lambd_params inside minimize method
        self.results = estimator.minimize(alpha_params.reshape(self.n_parameters), options=options, lambd_params=lambd_params)
        elapsed_time_sec = time.time() - start_time
        elapsed_time_str = elapsed_time_to_str(elapsed_time_sec)

        ## TODO: With nested configuration is not implemented yet
        if self.nested == True:
            final_log_likelihood = calcs.log_likelihood(self.results["alpha"], lambd = self.results["lambd"])
        else:
            final_log_likelihood = calcs.log_likelihood(self.results["alpha"])

        mcfadden_r2 = 1 - final_log_likelihood / log_likelihood_at_zero  # TODO: Implement a method to compute metrics

        # Store post-estimation information
        self.results["initial_log_likelihood"] = initial_log_likelihood
        self.results["final_log_likelihood"] = final_log_likelihood
        self.results["elapsed_time"] = elapsed_time_sec
        self.results["mcfadden_r2"] = mcfadden_r2
        self.results["pmle"] = pmle
        self.results["pmle_lamda"] = pmle_lambda
        self.results["method"] = method
        self.results["history"] = estimator.history

        if verbose >= 1:
            print("-------------------------------------------------------------------------\n"
                  "The kernel model has been estimated. Elapsed time: {elapsed_time}.\n"
                  "Final log-likelihood value: {final_log_likelihood:,.4f}\n"
                  "McFadden R^2: {r2:.4f}".format(elapsed_time=elapsed_time_str,
                                                  final_log_likelihood=final_log_likelihood,
                                                  r2 = mcfadden_r2))
            sys.stdout.flush()

        return None

    def summary(self) -> None:
        """Print a summary of the estimation results."""
        if self.results is None:
            msg = "The model has not been estimated yet. Use fit() to estimate it."
            logger_error(msg)
            raise RuntimeError(msg)

        print("-------------------------------------------------------------------------\n"
              "GKLR Kernel Model summary\n"
              "-------------------------------------------------------------------------\n"
              "Optimization method: {method}\n"
              "optimization success: {success}\n"
              "Optimization message: {message}\n"
              "Penalization: {pmle}\n"
              "Penalization parameter: {pmle_lambda}\n"
              "Initial log-likelihood: {initial_log_likelihood:,.4f}\n"
              "Final log-likelihood: {final_log_likelihood:,.4f}\n"
              "McFadden R^2: {r2:.4f}\n"
              "Elapsed time: {elapsed_time}\n"
              "-------------------------------------------------------------------------".format(
            method=self.results["method"],
            success=self.results["success"],
            message=self.results["message"],
            pmle=self.results["pmle"],
            pmle_lambda=self.results["pmle_lamda"],
            initial_log_likelihood=self.results["initial_log_likelihood"],
            final_log_likelihood=self.results["final_log_likelihood"],
            r2=self.results["mcfadden_r2"],
            elapsed_time=elapsed_time_to_str(self.results["elapsed_time"])))
        sys.stdout.flush()

        return None

    def predict_proba(self, train: bool = False) -> np.ndarray:
        """Predict class probabilities for the train or test kernel.

        Args:
            train: A boolean that indicates if the probability estimates belong to the training set (True) or test 
                set (False), only in the case that a test kernel matrix is defined. Default: False.

        Returns:
            Probability of the sample for each class in the model.
        """
        if self._K is None:
            msg = "Training kernel not found or not correctly defined. Use set_kernel_test() to compute it."
            logger_error(msg)
            raise RuntimeError(msg)

        if train and self.nested:
            # Create the Calcs instance for nests
            calcs = NestedKernelCalcs(K=self._K)
        elif train:
            # Create the Calcs instance
            calcs = KernelCalcs(K=self._K)
        else:
            if self._K_test is None:
                msg = "First you must compute the kernel for the test dataset using set_kernel_test()."
                logger_error(msg)
                raise RuntimeError(msg)
            if self.nested:
                # Create the Calcs instance for nests
                calcs = NestedKernelCalcs(K=self._K_test)
            else:
                # Create the Calcs instance
                calcs = KernelCalcs(K=self._K_test)

        if self.results is None:
            msg = "First you must estimate the model using fit()."
            logger_error(msg)
            raise RuntimeError(msg)

        if self.nested == True:
            proba = calcs.calc_probabilities(self.results["alpha"], lambd = self.results["lambda"])
        else:
            proba = calcs.calc_probabilities(self.results["alpha"])

        return proba

    def predict_log_proba(self, train: bool = False) -> np.ndarray:
        """Predict the natural logarithm of the class probabilities for the train or test kernel.

        Args:
            train: A boolean that indicates if the probability estimates belong to the training set (True) or test 
                set (False), only in the case that a test kernel matrix is defined. Default: False.

        Returns:
            Log-probability of the sample for each class in the model.
        """
        proba = self.predict_proba(train)
        return np.log(proba)

    def predict(self, train: bool = False) -> np.ndarray:
        """Predict class for the train or test kernel.
        
        Args:
            train: A boolean that indicates if the prediction belong to the training set (True) or test 
                set (False), only in the case that a test kernel matrix is defined. Default: False.

        Returns:
            Vector containing the class labels of the sample.
        """
        if self._K is None or self._K.alternatives is None:
            msg = "Training kernel not found or not correctly defined. Use set_kernel_test() to compute it."
            logger_error(msg)
            raise RuntimeError(msg)

        proba = self.predict_proba(train)
        encoded_labels = np.argmax(proba, axis=1)
        return self._K.get_alternatives().take(encoded_labels)

    def score(self) -> float | np.float64:
        """Predict the mean accuracy on the test kernel.

        Returns:
            Mean accuracy of `self.predict()`.
        """
        if self.choice_column is None:
            msg = "First you must compute the kernel for the train dataset using set_kernel_train()."
            logger_error(msg)
            raise RuntimeError(msg)

        if self._K_test is None or self._Z is None:
            msg = "First you must compute the kernel for the test dataset using set_kernel_test()."
            logger_error(msg)
            raise RuntimeError(msg)

        y_true = self._Z[self.choice_column]
        y_predict = self.predict()
        score = np.average(y_true == y_predict)
        return score
