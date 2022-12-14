"""GKLR optimizer module."""
from typing import Optional, Any, Dict, List, Union, Callable

import sys

import numpy as np
from scipy.optimize import OptimizeResult

from .kernel_utils import *
from .logger import *

class Optimizer():
    """Optimizer class object."""

    def __init__(self) -> None:
        """Constructor.
        """
        return

    def minimize(self,
                 fun: Callable,
                 x0: np.ndarray,
                 args: tuple = (),
                 method: str = "SGD",
                 jac: Optional[Union[Callable, bool]] = None,
                 hess: Optional[Callable] = None,
                 tol: float = 1e-06,
                 options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Minimize the objective function using the specified optimization method.

        Args:
            fun: The objective function to be minimized.
                ``fun(x, *args) -> float``,
                where ``x`` is the input vector and ``args`` are the additional
                arguments of the objective function.
            x0: The initial guess of the parameters.
            args: Additional arguments passed to the objective function.
            method: The optimization method. Default: "SGD".
            jac: The gradient of the objective function. Default: None.
            hess: The Hessian of the objective function. Default: None.
            tol: The tolerance for the termination. Default: 1e-06.
            options: A dictionary of solver options. Default: None.

        Returns:
            A dictionary containing the result of the optimization procedure:
                fun: The value of the objective function at the solution.
                x: A 1-D ndarray containing the solution.
                success: A boolean indicating whether the optimization converged.
                message: A string describing the cause of the termination.
        """
        if method is None:
            # Use the default method
            method = "SGD"

        if options is None:
            # Use the default options
            options = {}

        if tol is not None:
            options = dict(options)
            if method == 'SGD':
                options.setdefault('gtol', tol)

        if callable(jac):
            pass
        elif jac is True:
            # fun returns the objective function and the gradient
            fun = MemoizeJac(fun)
            jac = fun.derivative
        elif jac is None:
            jac = None
        else:
            # Default option if jac is not understood
            jac = None

        # TODO: Hessians are not implemented yet

        if method == "SGD":
            # Use the mini-batch gradient descent method
            res = self._minimize_mini_batch_sgd(fun, x0, jac=jac, args=args, **options)
        else:
            msg = (f"'method' = {method} is not a valid optimization method.\n"
                   f"Valid methods are: {CUSTOM_OPTIMIZATION_METHODS}.")
            logger_error(msg)
            raise ValueError(msg)

        return res

    def _minimize_mini_batch_sgd(self,
                                 fun: Callable,
                                 x0: np.ndarray,
                                 jac: Optional[Callable] = None,
                                 args: tuple = (),
                                 learning_rate: float = 1e-03,
                                 mini_batch_size: Optional[int] = None,
                                 n_samples: int = 0,
                                 gtol: float = 1e-06, 
                                 maxiter: int = 1000, # Number of epochs
                                 print_every: int = 0,
                                 seed: int = 0,
                                 **kwards,
    ) -> Dict[str, Any]:
        """Minimize the objective function using the stochastic gradient descent method.
        """

        # Checking errors
        if not callable(fun):
            m = "The objective function must be callable."
            logger_error(m)
            raise ValueError(m)
        if learning_rate <= 0:
            m = "The learning rate must be greater than zero."
            logger_error(m)
            raise ValueError(m)
        if mini_batch_size is None:
            # Use the entire dataset as the mini-batch (batch gradient descent)
            mini_batch_size = n_samples
        if mini_batch_size <= 0:
            m = "The mini-batch size must be greater than zero."
            logger_error(m)
            raise ValueError(m)
        if n_samples <= 0:
            m = ("The number of samples in the dataset must be greater than zero"
                 " and corresponds with number of rows in the dataset.")
            logger_error(m)
            raise ValueError(m)
        if mini_batch_size > n_samples:
            m = "The mini-batch size must be less than or equal to the number of samples in the dataset."
            logger_error(m)
            raise ValueError(m)
        if gtol <= 0:
            m = "The tolerance must be greater than zero."
            logger_error(m)
            raise ValueError(m)
        if maxiter <= 0:
            m = "The maximum number of iterations (epochs) must be greater than zero."
            logger_error(m)
            raise ValueError(m)
        if jac is None:
            # TODO: Implement the gradient-free optimization method using 2-point approximation
            m = "The gradient of the objective function must be provided."
            logger_error(m)
            raise ValueError(m)

        num_epochs = maxiter
        n, = x0.shape
        g = np.zeros((n,), np.float64)
        message = "Optimization terminated successfully."
        success = True

        # Optimization loop
        x = x0
        i = 0
        for i in range(num_epochs):
            # Define the random mini-batches. Increment the seed to reshuffle differently at each epoch
            seed += 1
            minibatches = self._random_mini_batch(n_samples, mini_batch_size, seed=seed)
            diff = np.zeros((n,), np.float64)
            loss_total = 0

            for minibatch in minibatches:
                # Compute the loss of the mini-batch if it is required
                if print_every > 0 and i % print_every == 0:
                    loss_total += fun(x, minibatch, *args)

                # Compute the gradient
                g = jac(x, minibatch, *args)
                diff = - learning_rate * g
                
                # Update the parameters
                x = x + diff

            if print_every > 0 and i % print_every == 0:
                loss_avg = loss_total / len(minibatches)
                print(f"\t* Epoch: {i}/{num_epochs} - Loss: {loss_avg:.4f}")
                sys.stdout.flush()
                
            if np.all(np.abs(diff) <= gtol):
                # Convergence
                message = "Optimization terminated successfully. Gradient tolerance reached."
                break

        i += 1
        if i >= num_epochs:
            message = 'STOP: TOTAL NO. of ITERATIONS REACHED LIMIT'
            success = False

        return OptimizeResult(x=x, fun=fun(x), jac=g, nit=i, nfev=i, 
            success=success, message=message)

    def _random_mini_batch(self,
                           n_samples: int,
                           mini_batch_size: int,
                           seed: int = 0,
    ) -> List[np.ndarray]:
        """
        Generate a list of random minibatches for the indices [0, ..., n_samples - 1]
        """
        np.random.seed(seed)
        indices = np.random.permutation(n_samples)
        mini_batches = []
        for i in range(0, n_samples, mini_batch_size):
            mini_batch = indices[i:i + mini_batch_size]
            mini_batches.append(mini_batch)
        return mini_batches


class MemoizeJac:
    """ Decorator that caches the return values of a function returning `(fun, grad)`
        each time it is called. """

    def __init__(self, fun):
        self.fun = fun
        self.jac = None
        self._value = None
        self.x = None
        self.minibatch = None

    def _compute_if_needed(self, x, minibatch = None,  *args):
        # Check if the function value has already been computed for the given x
        if not np.all(x == self.x) or self._value is None or self.jac is None:
            self.x = np.asarray(x).copy()
            if minibatch is not None:
                self.minibatch = np.asarray(minibatch).copy()
            else:
                self.minibatch = None
            fg = self.fun(x, minibatch, *args)
            self.jac = fg[1]
            self._value = fg[0]
        
        # Check if the mini-batches are the same as previous ones
        if not np.all(minibatch == self.minibatch):
            self.minibatch = np.asarray(minibatch).copy()
            fg = self.fun(x, minibatch, *args)
            self.jac = fg[1]
            self._value = fg[0]

    def __call__(self, x, *args):
        """ returns the the function value """
        self._compute_if_needed(x, *args)
        return self._value

    def derivative(self, x, *args):
        self._compute_if_needed(x, *args)
        return self.jac