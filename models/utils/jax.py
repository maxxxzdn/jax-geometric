import jax
import jax.numpy as jnp


def segment_mean(data, segment_ids, num_segments):
    """
    Computes the mean within segments of an array.
    """
    # Sum the data within each segment
    segment_sums = jax.ops.segment_sum(data, segment_ids, num_segments)
    # Compute the size of each segment
    segment_sizes = jax.ops.segment_sum(jnp.ones_like(data), segment_ids, num_segments)
    segment_means = segment_sums / segment_sizes
    return segment_means


def compute_radial(edge_index, x):
    """
    Compute x_i - x_j and ||x_i - x_j||^2.
    """
    senders, receivers = edge_index
    x_i, x_j = x[senders], x[receivers]
    distance = jnp.sum((x_i - x_j) ** 2, axis=1, keepdims=True)
    return distance
