"""GKLR nested kernel calculations module."""
from typing import List, Optional, Tuple
from scipy.special import logsumexp

import numpy as np

from .logger import *
from .kernel_matrix import KernelMatrix
from .kernel_utils import *
from .calcs import Calcs

class NestedKernelCalcs(Calcs):
    """Main calculations for the Nested Kernel Logistic Regression (NKLR) model."""
    
    def __init__(self, K: KernelMatrix, nests: List[List[int]], L0_cte: bool = False) -> None:
        """Constructor.

        Args:
            K: KernelMatrix object.
        """
        super().__init__(K)
        self.verbose = 0
        self.nests = nests
        self.lambd_shape = len(nests)
        self.mask = self.build_mask(self.lambd_shape, K.get_num_alternatives())
        self.mask_product = np.dot(self.mask.T,self.mask)
        self.group_of_alternatives = np.argmax(self.mask, axis=0)
        self.L0_cte = L0_cte
        self.nest_bool = self.mask.astype(bool)

        
    def build_mask(self, lambd_shape: int, num_alternatives: int) -> np.ndarray:
        """Build a mask for the nests.

        Args:
            lambd_shape: The number of nests.
            num_alternatives: The number of alternatives.

        Returns:
            A mask of shape (lambd_shape, num_alternatives) with 1 values
            for the alternatives in each nest and 0 otherwise.
        """
        mask = np.zeros((lambd_shape, num_alternatives), dtype=int)
        for nest_index, alts in enumerate(self.nests):
            mask[nest_index, alts] = 1
        return mask

    def calc_probabilities(self, 
                            alpha: np.ndarray,
                            lambd: np.ndarray, 
                            indices: Optional[np.ndarray] = None,
        ) -> np.ndarray:
            """Calculate the probabilities for each alternative.

            Obtain the probabilities for each alternative for each row of the
            dataset.

            Args:
                alpha: The vector of parameters. Shape: (num_cols_kernel_matrix, num_alternatives).
                lambd: The vector of nest dissimilarity parameters. Shape: (lambd_shape,).
                indices: The indices of the rows of the dataset for which the
                    probabilities are calculated. If None, the probabilities are
                    calculated for all rows of the dataset. Default: None.

            Returns:
                A matrix of probabilities for each alternative for each row of the
                    dataset. Each column corresponds to an alternative and each row
                    to a row of the dataset. The sum of the probabilities for each
                    row is 1. Shape: (n_samples, num_alternatives).
            """
            f = self.calc_f(alpha, indices=indices)
            Y = self.calc_Y(f, lambd)
            G, G_j = self.calc_G(Y, lambd)
            P = self.calc_P(Y, G, G_j)
            if self.verbose >= 2:
                print(f"Calculated f: {f}, Y: {Y}, G: {G}, G_j: {G_j}, P: {P}")
            
            return P

    def calc_Y(self,
               f: np.ndarray, 
               lambd: np.ndarray, 
               ) -> np.ndarray:
        """
        Computes the matrix Y whose columns are e^(f_j / lambda_K).
        
        Args:
            f: Utility matrix f of shape (n_samples, num_alternatives).
            lambd: The vector of nest dissimilarity parameters. Shape: (lambd_shape,).
            
        Returns:
            Matrix Y of shape (n_samples, num_alternatives).
        """
        lambd_per_alternative = lambd[self.group_of_alternatives].flatten()
        Y = np.exp(f / lambd_per_alternative)
        return Y

    def log_likelihood(self,
                       alpha: np.ndarray,
                       lambd: np.ndarray,
                       P: Optional[np.ndarray] = None,
                       choice_indices: Optional[np.ndarray] = None,
                       pmle: Optional[str] = None,
                       pmle_lambda: float = 0,
                       indices: Optional[np.ndarray] = None,
    ) -> float:
        """Calculate the log-likelihood of the NKLR model for the given parameters.

        Args:
            alpha: The vector of parameters. Shape: (num_cols_kernel_matrix, num_alternatives).
            lambd: The vector of nest dissimilarity parameters. Shape: (lambd_shape,).
            P: The matrix of probabilities of each alternative for each row of 
                the dataset. If None, the probabilities are calculated.
                Shape: (n_samples, num_alternatives). Default: None.
            choice_indices: The indices of the chosen alternatives for each row
                of the dataset. If None, the indices are obtained from the
                KernelMatrix object. Shape: (n_samples,). Default: None.
            pmle: It specifies the type of penalization for performing a penalized
                maximum likelihood estimation.  Default: None.
            pmle_lambda: The lambda parameter for the penalized maximum likelihood.
                 Default: 0.
            indices: The indices of the rows of the dataset for which the
                log-likelihood is calculated. If None, the log-likelihood is
                calculated for all rows of the dataset. Default: None.

        Returns:
            The log-likelihood of the NKLR model for the given parameters.
        """
        if indices is None:
            num_rows = self.K.get_num_rows()
        else:
            num_rows = indices.shape[0]
        if P is None:
            P = self.calc_probabilities(alpha, indices=indices, lambd=lambd)
        else:
            if P.shape != (num_rows, self.K.get_num_alternatives()):
                m = (f"P has {P.shape} dimensions, but it should have "
                    f" dimensions: ({num_rows}, {self.K.get_num_alternatives()}).")
                logger_error(m)
                raise ValueError(m)
        if choice_indices is None:
            choice_indices = self.K.get_choices_indices()
            if indices is not None:
                indices = indices.tolist()
                choice_indices = choice_indices[indices]
        else:
            if len(choice_indices) != P.shape[0]:
                m = (f"choice_indices has {len(choice_indices)} elements, but P"
                    f" has {P.shape[0]} rows.")
                logger_error(m)
                raise ValueError(m)

        # Compute the log-likelihood from the matrix of probabilities
        log_P = np.log(P)
        log_likelihood = np.sum(log_P[np.arange(len(log_P)), choice_indices]) 
        log_likelihood /= num_rows

        # Compute the penalty function
        penalty = 0
        if pmle is None:
            pass
        elif pmle == "Tikhonov":
            penalty = self.tikhonov_penalty(alpha, pmle_lambda, indices=indices)
        else:
            msg = f"'pmle' = {pmle} is not a valid value for the penalization."
            logger_error(msg)
            raise ValueError(msg)

        return log_likelihood + penalty

    def gradient(self,
                 alpha: np.ndarray,
                 lambd: np.ndarray,
                 P: Optional[np.ndarray] = None,
                 pmle: Optional[str] = None,
                 pmle_lambda: float = 0,
                 indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate the gradient of the log-likelihood function of the NKLR model 
        for the given parameters.

        Args:
            alpha: The vector of parameters. Shape: (num_cols_kernel_matrix, num_alternatives).
            lambd: The vector of nest dissimilarity parameters. Shape: (lambd_shape,).
            pmle: It specifies the type of penalization for performing a penalized
                maximum likelihood estimation.  Default: None.
            pmle_lambda: The lambda parameter for the penalized maximum likelihood.
                 Default: 0.
            P: The matrix of probabilities of each alternative for each row of 
                the dataset. If None, the probabilities are calculated.
                Shape: (n_samples, num_alternatives). Default: None.
            indices: The indices of the rows of the dataset for which the
                log-likelihood is calculated. If None, the log-likelihood is
                calculated for all rows of the dataset. Default: None.

        Returns:
            The gradient of the log-likelihood function of the NKLR model for the
            given parameters. Shape: (num_rows_kernel_matrix * num_alternatives, ).
        """
        if indices is None:
            num_rows = self.K.get_num_rows()
        else:
            num_rows = indices.shape[0]

        if P is None:
            f = self.calc_f(alpha, indices=indices)
            Y = self.calc_Y(f, lambd)
            G, G_j = self.calc_G(Y, lambd)
            P = self.calc_P(Y, G, G_j)
        else:
            if P.shape != (num_rows, self.K.get_num_alternatives()):
                raise ValueError(f"P has shape {P.shape}, expected ({num_rows}, {self.K.get_num_alternatives()})")

            # Recompute f and Y to keep the cached probabilities consistent with the current parameters.
            f = self.calc_f(alpha, indices=indices)
            Y = self.calc_Y(f, lambd)

            if Y.shape != (num_rows, self.K.get_num_alternatives()):
                raise ValueError(f"Y has shape {Y.shape}, expected ({num_rows}, {self.K.get_num_alternatives()})")


        # Compute the gradient of the log-likelihood function.
        N = P.shape[0]
        n_alts = self.K.get_num_alternatives()
        D_k = np.dot(Y, self.mask.T)
        P_cond = Y / D_k[:, self.group_of_alternatives]

        Z = self.K.get_choices_matrix()
        if indices is not None:
            Z = Z[indices, :]

        alt_chosen = np.argmax(Z, axis=1)
        chosen_nests = self.group_of_alternatives[alt_chosen]

        Z_nest = np.zeros_like(D_k)
        Z_nest[np.arange(N), chosen_nests] = 1

        Pk = np.dot(P, self.mask.T)

        # --- Gradient with respect to alpha ---
        lambd_per_alternative = lambd[self.group_of_alternatives].flatten()
        dL_df = (Z / lambd_per_alternative) \
            + ((1.0 - 1.0 / lambd_per_alternative) 
               * Z_nest[:, self.group_of_alternatives] * P_cond) \
               - P

        alpha_gradient = np.zeros((self.K.get_num_cols(), n_alts), dtype=DEFAULT_DTYPE)
        for alt in range(n_alts):
            alpha_gradient_alt = self.K.dot(dL_df[:, alt], K_index=alt, col_indices=indices)
            alpha_gradient[:, alt] = (alpha_gradient_alt / N).reshape((self.K.get_num_cols(),))

        if pmle is None:
            pass
        elif pmle == "Tikhonov":
            alpha_gradient += self.tikhonov_penalty_gradient(alpha, pmle_lambda, indices=indices)
        else:
            msg = f"ERROR. {pmle} is not a valid value for the penalization method `pmle`."
            logger_error(msg)
            raise ValueError(msg)
        
        # --- Gradient with respect to lambda ---
        E_f_in_k = np.zeros((N, self.lambd_shape), dtype=DEFAULT_DTYPE)
        for k in range(self.lambd_shape):
            idx_k = (self.group_of_alternatives == k)
            if np.any(idx_k):
                E_f_in_k[:, k] = (P_cond[:, idx_k] * f[:, idx_k]).sum(axis=1)

        T = np.log(D_k) - (E_f_in_k / lambd)
        lambd_gradient_by_sample = (Z_nest - Pk) * T
        lambd_gradient_by_sample[np.arange(N), chosen_nests] += \
            -(f[np.arange(N), alt_chosen] - E_f_in_k[np.arange(N), chosen_nests]) \
            / (lambd[chosen_nests] ** 2)
        lambd_gradient = lambd_gradient_by_sample.mean(axis=0)

        if self.L0_cte:
            lambd_gradient[0] = 0.0

        if self.verbose >= 2:
            print(f"lambd: {lambd}")
            print(f"lambd_gradient: {lambd_gradient}")
            print(f"∥grad∥ (lambd): {np.linalg.norm(lambd_gradient)}")

        return np.concatenate((lambd_gradient, alpha_gradient.ravel()))

    def calc_f(self, 
              alpha: np.ndarray, 
              indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate the value of utility function for each alternative for each row
        of the dataset.

        Args:
            alpha: The vector of parameters. Shape: (num_cols_kernel_matrix, num_alternatives).
            indices: The indices of the rows of the dataset for which the utility
                function is calculated. If None, all the rows are used. Default: None.

        Returns:
            A matrix where each row corresponds to the utility of each alternative
                for each row of the dataset. Shape: (n_samples, num_alternatives).
        """
        num_rows = self.K.get_num_rows()
        if indices is not None:
            if np.max(indices) >= self.K.get_num_rows() or np.min(indices) < 0:
                msg = "Some of the indices provided to compute utility function are out of range."
                logger_error(msg)
                raise ValueError(msg)
            else:
                num_rows = indices.shape[0]

        n_alts = self.K.get_num_alternatives()
        f = np.zeros((num_rows, n_alts), dtype=DEFAULT_DTYPE)
        for alt in range(0,n_alts):
            alpha_alt = alpha[:, alt].reshape(self.K.get_num_cols(),1)  # Get only the column for alt
            f_alt = self.K.dot(alpha_alt, K_index=alt, row_indices=indices)
            f[:, alt] = f_alt.reshape((num_rows,))  # Store the result in the corresponding column
        
        if self.verbose >= 2:
            print("f shape:", f.shape)
            print("f mean:", np.mean(f), "std:", np.std(f))

            print("f[0]:", f[0])
            print("alpha[:, 0]:", alpha[:, 0])
            print("K0.dot(alpha[:,0]):", self.K.dot(alpha[:, 0], K_index=0))

        return f

    def calc_G(self,
                Y : np.ndarray, 
                lambd: np.ndarray, 
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the generating function `G` of a Generalized Extreme Value
            (GEV) model and its derivative. For NKLR model, the generating function is computed by first summing
            the auxiliary terms Y_ij within each nest, raising each nest sum to its
            corresponding dissimilarity parameter lambda_k, and then summing the
            resulting nest-level terms across all nests.
        
        Args:
            Y: The auxiliary matrix that contains the exponentiated utility values scaled
                by the dissimilarity parameter of the corresponding nest. Shape: (n_samples, num_alternatives).
            lambd: The vector of nests parameters. Shape: (lambd_shape,).

        Returns:
             A tuple with the generating function G. Shape: (n_samples, 1)
                 and its derivative G_j. Shape: (n_samples, num_alternatives).
        """
        D_k = np.dot(Y, self.mask.T)
        G = np.sum(D_k ** lambd.T, axis=1).reshape((Y.shape[0], 1))
        G_j = np.dot(D_k ** (lambd.T - 1), self.mask)

        return (G, G_j)

    def tikhonov_penalty(self,
                         alpha: np.ndarray,
                         pmle_lambda: float,
                         indices: Optional[np.ndarray] = None,
    ) -> float:
        """Calculate the Tikhonov penalty for the given parameters.

        Args:
            alpha: The vector of parameters. Shape: (num_cols_kernel_matrix, num_alternatives).
            pmle_lambda: The lambda parameter for the penalized maximum likelihood.

        Returns:
            The Tikhonov penalty for the given parameters.
        """
        pen_raw = 0.0
        for alt in range(self.K.get_num_alternatives()):
            alpha_alt = alpha[:, alt:alt+1]
            pen_raw += float(alpha_alt.T @ self.K.dot(alpha_alt, K_index=alt))
        num_rows = self.K.get_num_rows() if indices is None else len(indices)
        return -0.5 * pmle_lambda * pen_raw / num_rows

    def tikhonov_penalty_gradient(self,
                                  alpha: np.ndarray,
                                  pmle_lambda: float,
                                  indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate the gradient of the Tikhonov penalty for the given parameters.

        Args:
            alpha: The vector of parameters. Shape: (num_cols_kernel_matrix, num_alternatives).
            pmle_lambda: The lambda parameter for the penalized maximum likelihood.
            indices: The indices of the rows of the dataset for which the gradient

        Returns:
            The gradient of the Tikhonov penalty for the given parameters.
                If indices is None, the shape is (num_cols_kernel_matrix, num_alternatives),
                otherwise, the shape is (len(indices), num_alternatives).
        """
        num_rows = self.K.get_num_rows() if indices is None else len(indices)
        grad = np.zeros_like(alpha)
        for alt in range(self.K.get_num_alternatives()):
            alpha_alt = alpha[:, alt:alt+1]
            grad[:, alt] = (-pmle_lambda * self.K.dot(alpha_alt, K_index=alt) / num_rows).ravel()
        return grad
      