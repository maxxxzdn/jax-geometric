import torch
import torch.nn.functional as F
from torch import nn

from .algebra.cliffordalgebra import CliffordAlgebra
from .modules.gp import SteerableGeometricProductLayer
from .modules.linear import MVLinear
from .modules.mvlayernorm import MVLayerNorm
from .modules.mvsilu import MVSiLU



def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class CEMLP(nn.Module):
    def __init__(
        self,
        algebra,
        in_features,
        hidden_features,
        out_features,
        n_layers=2,
        normalization_init=0,
    ):
        super().__init__()
        self.algebra = algebra
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.n_layers = n_layers


        layers = []

        # Add geometric product layers.
        for i in range(n_layers - 1):
            layers.append(
                nn.Sequential(
                    MVLinear(self.algebra, in_features, hidden_features),
                    MVSiLU(self.algebra, hidden_features),
                    SteerableGeometricProductLayer(
                        self.algebra,
                        hidden_features,
                        normalization_init=normalization_init,
                    ),
                    MVLayerNorm(self.algebra, hidden_features),
                )
            )
            in_features = hidden_features

        # Add final layer.
        layers.append(
            nn.Sequential(
                MVLinear(self.algebra, in_features, out_features),
                MVSiLU(self.algebra, out_features),
                SteerableGeometricProductLayer(
                    self.algebra,
                    out_features,
                    normalization_init=normalization_init,
                ),
                MVLayerNorm(self.algebra, out_features),
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class EGCL(nn.Module):
    def __init__(
        self,
        algebra,
        in_features,
        hidden_features,
        out_features,
        edge_attr_features=0,
        node_attr_features=0,
        residual=True,
        normalization_init=0,
    ):
        super().__init__()
        self.residual = residual
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.edge_attr_features = edge_attr_features
        self.node_attr_features = node_attr_features

        self.edge_model = CEMLP(
            algebra,
            self.in_features + self.edge_attr_features,
            self.hidden_features,
            self.out_features,
            normalization_init=normalization_init,
        )

        self.node_model = CEMLP(
            algebra,
            self.in_features + self.out_features,
            self.hidden_features,
            self.out_features,
            normalization_init=normalization_init,
        )

    def message(self, h_i, h_j, edge_attr=None):
        if edge_attr is None:  # Unused.
            input = h_i - h_j
        else:
            input = torch.cat([h_i - h_j, edge_attr], dim=1)

        h_msg = self.edge_model(input)
        return h_msg

    def aggregate(self, h_msg, segment_ids, num_segments):
        h_agg = unsorted_segment_mean(h_msg, segment_ids, num_segments=num_segments)
        return h_agg

    def update(self, h_agg, h, node_attr=None):
        if node_attr is not None:
            input_h = torch.cat([h, h_agg, node_attr], dim=1)
        else:
            input_h = torch.cat([h, h_agg], dim=1)

        out_h = self.node_model(input_h)

        if self.residual:
            out_h = h + out_h

        return out_h

    def forward(self, h, edge_index, edge_attr=None, node_attr=None):
        # Message
        rows, cols = edge_index

        h_i, h_j = h[rows], h[cols]

        h_msg = self.message(h_i, h_j, edge_attr)

        # Aggregate
        agg_h = self.aggregate(h_msg.flatten(1), rows, num_segments=len(h)).view(
            len(h), *h_msg.shape[1:]
        )

        # Update
        h = self.update(agg_h, h, node_attr)

        return h


class CEGNN(nn.Module):
    def __init__(
        self,
        in_features=3,
        hidden_features=28,
        out_features=1,
        edge_features_in=1,
        n_layers=3,
        normalization_init=0,
        residual=True,
    ):
        super().__init__()
        self.algebra = CliffordAlgebra((1.0, 1.0, 1.0))
        self.hidden_features = hidden_features
        self.n_layers = n_layers

        self.embedding = MVLinear(
            self.algebra, in_features, hidden_features, subspaces=False
        )

        layers = []

        for i in range(0, n_layers):
            layers.append(
                EGCL(
                    self.algebra,
                    hidden_features,
                    hidden_features,
                    hidden_features,
                    edge_features_in,
                    residual=residual,
                    normalization_init=normalization_init,
                )
            )

        self.projection = nn.Sequential(
            MVLinear(self.algebra, hidden_features, out_features),
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, h, edges, edge_attr):
        h = self.embedding(h)
        for layer in self.layers:
            h = layer(h, edges, edge_attr=edge_attr)

        h = self.projection(h)
        return h