""" Adapted to JAX from https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks/blob/master/models/modules/fcgp.py"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import normal

from typing import Tuple


class MVLinear(nn.Module):
    """
    Equivariant linear transformation of multivectors.

    Attributes:
        algebra (CliffordAlgebra): An instance of CliffordAlgebra defining the algebraic structure.
        features (int): The number of output features.
        bias_dims (Tuple[int, ...]): Dimensions for the bias terms.
        subspaces (bool): If True, applies the transformation across subspaces. Default is True.
    """

    algebra: object
    features: int
    bias_dims: Tuple[int, ...] = (0,)
    subspaces: bool = True

    @nn.compact
    def __call__(self, input):
        """
        Apply the linear transformation to each grade of the input multivector.

        Args:
            input (jnp.ndarray): multivector of shape (batch_size, in_features, ..., 2**algebra.dim).

        Returns:
            output (jnp.ndarray): multivector of shape (batch_size, out_features, ..., 2**algebra.dim).
        """
        in_features = jnp.shape(input)[1]
        
        # Initializing the weights
        stddev = 1 / jnp.sqrt(in_features)
        weight_shape = (
            (self.features, in_features, self.algebra.n_subspaces)
            if self.subspaces
            else (self.features, in_features)
        )
        weight = self.param("weight", normal(stddev), weight_shape)

        # Forward pass for each grade or whole multivector
        if self.subspaces:
            weight = jnp.repeat(weight, self.algebra.subspaces, axis=-1)
            result = jnp.einsum("bm...i, nmi->bn...i", input, weight)
        else:
            result = jnp.einsum("bm...i, nm->bn...i", input, weight)

        # Defining and initializing bias if required
        if self.bias_dims is not None:
            bias_shape = (1, self.features, len(self.bias_dims))
            bias = self.param("bias", normal(0.1 * stddev), bias_shape)
            # Broadcast bias across the batch dimensions
            bias = self.algebra.embed(bias, self.bias_dims)
            result += jax.lax.broadcast_in_dim(
                bias, result.shape, (0, 1, len(result.shape) - 1)
            )

        return result
