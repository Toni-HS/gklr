"""GKLR kernel_estimator module."""
from cmath import log
from typing import Optional, Tuple, Any, Dict

from asyncio.log import logger
import sys

import numpy as np

from .nested_kernel_calcs import NestedKernelCalcs
from .logger import *
from .kernel_utils import *
from .estimation import Estimation

class NestedKernelEstimator(Estimation):
    """Estimation object for the Nested Kernel Logistic Regression (NKLR) model."""

    def __init__(self,
                 calcs: NestedKernelCalcs,
                 pmle: Optional[str] = None,
                 pmle_lambda: float = 0.0,
                 method: str = "L-BFGS-B",
                 verbose: int = 1,
    ) -> None:
        """Constructor.

        Args:
            calcs: Calcs object.
            pmle: Indicates the penalization method for the penalized maximum
                likelihood estimation. If 'None' a maximum likelihood estimation
                without penalization is performed. Default: None.
            pmle_lambda: The value of the regularization parameter for the PMLE
                method. Default: 0.0.
            method: The optimization method. Default: "L-BFGS-B".
            verbose: Indicates the level of verbosity of the function. If 0, no
                output will be printed. If 1, basic information about the
                estimation procedure will be printed. If 2, the information
                about each iteration will be printed. Default: 1.
        """
        if pmle not in VALID_PMLE_METHODS:
            msg = (f"'pmle' = {pmle} is not a valid value for the penalization"
                   f" method. Valid methods are: {VALID_PMLE_METHODS}.")
            logger_error(msg)
            raise ValueError(msg)

        super().__init__(calcs, pmle, pmle_lambda, method, verbose)
        self.calcs = calcs
        self.alpha_shape = (calcs.K.get_num_cols(), calcs.K.get_num_alternatives())
        self.lambd_shape = calcs.lambd_shape
        self.P_cache = None # Cache for the matrix of probabilities P
        self.Y_cache = None # Cache for the matrix of auxiliary values Y
        self.prev_alpha_params = None # Previous parameters used in the objective function
        self.prev_lambd_params = None # Previous nest parameters used in the objective function
        self.prev_indices = None # Previous indices used in the objective function
        self.lambda_low  = 0.15
        self.lambda_high = 1.0

    def _build_bounds(self):
        """Build the optimization bounds for the NKLR parameter vector.
        
            Returns:
                list[tuple[float | None, float | None]]: Bounds for the optimizer.
                The first lambd_shape entries correspond to lambda parameters and
                the remaining entries correspond to alpha parameters.
            """
        n_alpha = np.prod(self.alpha_shape)
        return ([(self.lambda_low, self.lambda_high)] * self.lambd_shape +
                [(None, None)] * n_alpha)

    def objective_function(self,
                           params: np.ndarray,
                           indices: Optional[np.ndarray] = None
    ) -> float:
        """Compute the objective function for the Nested Kernel Logistic Regression 
        (NKLR) model and its gradient.

        Args:
            params: The model parameters. Shape: (lambd_shape + n_params,).
            indices: The indices of the samples to be used in the computation of
                the objective function. If 'None' all the samples will be used.
                Default: None.
        Returns:
            A tuple with the value of the objective function and its gradient.
            The first element of the tuple is the value of the objective function
            and the second element is the gradient of the objective function with 
            respect to the model parameters with shape: (num_rows_kernel_matrix * num_alternatives,)
        """
        # Convert params to lambd_params and alpha_params and reshape them as a column vector
        lambd_params = params[:self.lambd_shape]
        alpha_params = params[self.lambd_shape:].reshape(self.alpha_shape)

        if self.prev_alpha_params is None or self.prev_lambd_params is None or \
            not np.array_equal(alpha_params, self.prev_alpha_params) or \
            not np.array_equal(lambd_params, self.prev_lambd_params) or \
            (indices is not None and self.prev_indices is None) or \
            (indices is None and self.prev_indices is not None) or \
            (indices is not None and self.prev_indices is not None and \
            not np.array_equal(indices, self.prev_indices)):
            # Compute the matrix of probabilities P and store it in the cache
            P = self.calcs.calc_probabilities(alpha_params, indices=indices, lambd=lambd_params)
            self.P_cache = P
            self.prev_alpha_params = alpha_params
            self.prev_lambd_params = lambd_params
            self.prev_indices = indices
        else:
            # Reuse the cached matrix of probabilities P
            P = self.P_cache

        # Compute the log-likelihood
        ll = self.calcs.log_likelihood(alpha_params, P=P, pmle=self.pmle, pmle_lambda=self.pmle_lambda, indices=indices, lambd=lambd_params)
        self.history["loss"].append(-ll)

        if self.verbose >= 2:
            print(f"Current objective function: {-ll:,.4f}", end = "\r")
            sys.stdout.flush()
        return (-ll)

    def gradient(self,
                 params: np.ndarray,
                 indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the gradient of the objective function for the Nested Kernel Logistic
        Regression (NKLR) model.
        
        This function is used by the optimization methods that do not require
        the computation of the objective function. If the objective function is
        also required, it is more efficient to use the 'objective_function'
        method, setting the 'return_gradient' argument to 'True'.

        Args:
            params: The model parameters. Shape: (lambd_shape + n_params,).
            indices: The indices of the samples to be used in the computation of
                the the gradient. If 'None' all the samples will be used.
                Default: None.
        
        Returns:
            The gradient of the objective function with respect to the model
            parameters with shape: (lambd_shape + num_rows_kernel_matrix * num_alternatives,).
        """
        # Convert params to lambd_params and alpha_params and reshape them as a column vector
        lambd_params = params[:self.lambd_shape]
        alpha_params = params[self.lambd_shape:].reshape(self.alpha_shape)

        if self.prev_alpha_params is None or self.prev_lambd_params is None or \
            not np.array_equal(alpha_params, self.prev_alpha_params) or \
            not np.array_equal(lambd_params, self.prev_lambd_params) or \
            (indices is not None and self.prev_indices is None) or \
            (indices is None and self.prev_indices is not None) or \
            (indices is not None and self.prev_indices is not None and \
            not np.array_equal(indices, self.prev_indices)):
            # Compute the matrix of probabilities P and store it in the cache
            f = self.calcs.calc_f(alpha_params, indices=indices)
            Y = self.calcs.calc_Y(f, lambd_params)
            G, G_j = self.calcs.calc_G(Y, lambd_params)
            P = self.calcs.calc_P(Y, G, G_j)
            self.P_cache = P
            self.prev_alpha_params = alpha_params
            self.prev_lambd_params = lambd_params
            self.prev_indices = indices
        else:
            # Reuse the cached matrix of probabilities P
            P = self.P_cache

        # Compute the log-likelihood and gradient
        gradient = self.calcs.gradient(
            alpha_params, 
            lambd_params, 
            P=P, 
            pmle=self.pmle, 
            pmle_lambda=self.pmle_lambda, 
            indices=indices
        )

        if self.verbose >= 2:
            lambd_grad = gradient[:self.lambd_shape]
            print(f"Lambda gradient: {lambd_grad}, norm: {np.linalg.norm(lambd_grad):.6f}")
            sys.stdout.flush()
        return -gradient

    def objective_function_with_gradient(self,
                                         params: np.ndarray,
                                         indices: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray]:
        """Compute the objective function for the Nested Kernel Logistic Regression 
        (NKLR) model and its gradient.

        Args:
            params: The model parameters. Shape: (lambd_shape + n_params,).
            indices: The indices of the samples to be used in the computation of
                the objective function. If 'None' all the samples will be used.
                Default: None.
        Returns:
            A tuple with the value of the objective function and its gradient.
            The first element of the tuple is the value of the objective function
            and the second element is the gradient of the objective function with 
            respect to the model parameters with shape: (num_rows_kernel_matrix * num_alternatives,)
        """
        # Compute the log-likelihood and gradient
        obj = self.objective_function(params, indices=indices)
        gradient = self.gradient(params, indices=indices)
        return (obj, gradient)


    def minimize(self,
                 alpha_params: np.ndarray,
                 lambd_params: np.ndarray,
                 loss_tol: float = 1e-06,
                 options: Optional[Dict[str, Any]] = None,
                 **kargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Minimize the objective function.

        Args:
            alpha_params: The initial values of the model parameters. Shape: (n_params,).
            lambd_params: The initial values of the nest parameters. Shape: (lambd_shape,).
            loss_tol: The tolerance for the loss function. Default: 1e-06.
            options: A dict with advanced options for the optimization method. 
                Default: None.
            **kargs: Additional arguments for the minimization function.

        Returns:
            A dict with the results of the optimization.
        """
        params = np.concatenate((lambd_params, alpha_params))
        bounds = self._build_bounds()
 
        results = super().minimize(params, loss_tol, options, bounds=bounds, lambd_shape=self.lambd_shape, **kargs)
        # Convert alpha_params to alpha np vector and reshape them as a column vector
        results["lambd_params"] = results["params"][:self.lambd_shape]
        results["alpha_params"] = results["params"][self.lambd_shape:].reshape(self.alpha_shape)
        return results