import jax.numpy as jnp
import flax.linen as nn

from ...utils.jax import segment_mean
from ...cegnn.jax.algebra.cliffordalgebra import CliffordAlgebra
from ...cegnn.jax.modules.linear import MVLinear
from ...cegnn.jax.modules.activation import MVSiLU
from ...cegnn.jax.modules.norm import MVLayerNorm
from ...cegnn.jax.modules.gp import SteerableGeometricProductLayer


class CEMLP(nn.Module):
    algebra: CliffordAlgebra
    hidden_features: int
    out_features: int
    product_paths_sum: int
    n_layers: int = 2
    normalization: bool = True
    
    @nn.compact
    def __call__(self, x):
        for idx in range(self.n_layers):
            
            out_features = self.hidden_features if idx < self.n_layers - 1 else self.out_features
             
            # single layer
            x = MVLinear(self.algebra, out_features)(x)
            x = MVSiLU(self.algebra)(x)
            x = SteerableGeometricProductLayer(self.algebra, self.hidden_features, self.product_paths_sum, normalization=self.normalization)(x)
            x = MVLayerNorm(self.algebra)(x)
            
        return x
    
    
def build_fn(algebra, hidden_dim, normalization, product_paths_sum):
    """
    CEGNN primitives as functions
        1. message function
        2. message aggregation + node update
        3. message aggregation + positional update
    """

    def message_fn(edge_index, h, edge_attr):
        """
        Message: m_ij = phi_e(h_i^l - h_j^l, a_ij)
        """
        phi_e = CEMLP(
            algebra=algebra, 
            hidden_features=hidden_dim, 
            out_features=hidden_dim, 
            product_paths_sum=product_paths_sum,
            n_layers=2, 
            normalization=normalization
        )

        senders, receivers = edge_index
        h_i, h_j = h[senders], h[receivers]
        out = jnp.concatenate([h_i - h_j, edge_attr], axis=1)

        return phi_e(out)

    def agg_update_fn(edge_index, h_i, m_ij):
        """
        Aggregation: m_i = sum_{j!=i} m_ij

        Node update: h_i^{l+1} = phi_h(h_i^l, m_i)
        """
        phi_h = CEMLP(
            algebra=algebra, 
            hidden_features=hidden_dim, 
            out_features=hidden_dim, 
            product_paths_sum=product_paths_sum,
            n_layers=2, 
            normalization=normalization
        )

        senders, _ = edge_index
        m_i = segment_mean(m_ij, senders, num_segments=h_i.shape[0])
        out = jnp.concatenate([h_i, m_i], axis=1)

        return h_i + phi_h(out)

    return message_fn, agg_update_fn


class CEGNN_layer(nn.Module):
    algebra: CliffordAlgebra
    features: int
    product_paths_sum: int
    normalization: bool = True
    
    @nn.compact
    def __call__(self, h, edge_index, edge_attr=None):
        """
        CEGNN layer
        """
        message_fn, agg_update_fn = build_fn(
            algebra=self.algebra, 
            hidden_dim=self.features, 
            normalization=self.normalization,
            product_paths_sum=self.product_paths_sum,
        )
        
        # Message
        m_ij = message_fn(edge_index, h, edge_attr)
        
        # Aggregate + Update
        h = agg_update_fn(edge_index, h, m_ij)
        
        return h
    
    
class CEGNN(nn.Module):
    algebra: CliffordAlgebra
    hidden_features: int
    out_features: int
    n_layers: int
    product_paths_sum: int
    normalization: int = True
    
    @nn.compact
    def __call__(self, h, edge_index, edge_attr):
        """
        CEGNN model
        """
    
        # Embedding
        h = MVLinear(self.algebra, self.hidden_features, subspaces=False)(h)
        
        for _ in range(self.n_layers):
            h = CEGNN_layer(
                algebra=self.algebra, 
                features=self.hidden_features, 
                product_paths_sum=self.product_paths_sum,
                normalization=self.normalization
            )(h, edge_index, edge_attr)
        
        # Projection
        h = MVLinear(self.algebra, self.out_features)(h)
        
        return h