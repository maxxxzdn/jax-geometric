""" Adapted to JAX from https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks/blob/master/models/modules/fcgp.py"""
import jax
import jax.numpy as jnp
from flax import linen as nn

from .norm import GradeNorm
from .linear import MVLinear
from .cayley import WeightedCayleyElementWise

from typing import Tuple


class SteerableGeometricProductLayer(nn.Module):
    """
    Element-wise GP layer using steerable geometric products in Clifford algebra.

    This layer combines equivariant linear transformations and weighted geometric product.
    The difference with the FullyConnectedSteerableGeometricProductLayer is that this layer preserves the number of input features.

    Attributes:
        algebra (CliffordAlgebra): An instance of CliffordAlgebra defining the algebraic structure.
        features (int): The number of input features.
        bias_dims (Tuple[int, ...]): Dimensions for the bias terms.
        include_first_order (bool): Whether to compute the linear term in the output. Default is False.
        normalization (bool): Whether to apply grade-wise normalization to the inputs. Default is True.
    """

    algebra: object
    features: int
    product_paths_sum: int
    bias_dims: Tuple[int, ...] = (0,)
    include_first_order: bool = True
    normalization: bool = True

    @nn.compact
    def __call__(self, input):
        """
        Defines the computation performed at every call.

        Args:
            input: The input tensor to the layer.

        Returns:
            The output tensor of the fully connected steerable geometric product layer.
        """

        # Creating a weighted Cayley transform
        weighted_cayley = WeightedCayleyElementWise(
            self.algebra, self.features, self.product_paths_sum
        )()

        # Applying a linear transformation to the input
        input_right = MVLinear(
            self.algebra, self.features, bias_dims=None
        )(input)

        if self.normalization:
            input_right = GradeNorm(self.algebra)(input_right)

        # Weighted geometric product
        out = jnp.einsum(
            "bn...i, nijk, bn...k -> bn...j", input, weighted_cayley, input_right
        )

        if self.include_first_order:
            # Adding the linear term
            out += MVLinear(
                self.algebra,
                self.features,
                bias_dims=self.bias_dims,
            )(input)
            out = out / jnp.sqrt(2)

        return out
