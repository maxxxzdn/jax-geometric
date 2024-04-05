""" Inspired by https://github.com/Qualcomm-AI-research/geometric-algebra-transformer/blob/main/gatr/primitives/nonlinearities.py"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import zeros, ones


class MVGELU(nn.Module):

    def __call__(self, input):
        """
        Multivector adaptation of the GELU activation function.
        GELU is computed for the scalar part of the input and used as a gate for the multivector input:

            out = GELU(input[0]) * input

        Args:
            input (jnp.ndarray): multivector of shape (..., 2**algebra.dim).

        Returns:
            out (jnp.ndarray): multivector of shape (..., 2**algebra.dim).
        """
        scalar = input[..., [0]]
        gates = jax.nn.sigmoid(
            jnp.sqrt(2 / jnp.pi) * (2 * (scalar + 0.044715 * jnp.power(scalar, 3)))
        )
        return gates * input


class MVSiLU(nn.Module):
    
    algebra: object
    
    @nn.compact
    def __call__(self, input):
        """
        Multivector adaptation of the SiLU activation function.

        Args:
            input (jnp.ndarray): multivector of shape (..., 2**algebra.dim).

        Returns:
            out (jnp.ndarray): multivector of shape (..., 2**algebra.dim).
        """
        factor1 = self.param("factor1", ones, (1, input.shape[1], self.algebra.dim + 1))
        factor2 = self.param("factor2", zeros, (1, input.shape[1], self.algebra.dim + 1))
        
        norms = self.algebra.qs(input, grades=self.algebra.grades[1:])
        norms = jnp.concatenate([input[..., [0]], *norms], axis=-1)
        
        norms = factor1 * norms + factor2
        norms = jnp.repeat(norms, self.algebra.subspaces, axis=-1)
        
        return input * jax.nn.sigmoid(norms)