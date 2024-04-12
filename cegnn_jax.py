import argparse
import jax
import jax.numpy as jnp
import time
import numpy as np
import tqdm

from models.cegnn.jax.cegnn import CEGNN
from models.cegnn.jax.algebra.cliffordalgebra import CliffordAlgebra
from benchmarks.utils import get_edges_batch_jax


parser = argparse.ArgumentParser()
parser.add_argument('--hidden_features', type=int, default=32)
parser.add_argument('--edge_features', type=int, default=1)
parser.add_argument('--out_features', type=int, default=1)
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--normalization', type=int, default=1)

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_nodes', type=int, default=5)

parser.add_argument('--n_warmup', type=int, default=10)
parser.add_argument('--n_iter', type=int, default=100)
parser.add_argument('--n_measurements', type=int, default=10)

args = parser.parse_args()

durations = []

for measurement in tqdm.tqdm(range(args.n_measurements)):
    key = jax.random.PRNGKey(42 + measurement) 
    algebra = CliffordAlgebra((1,1,1))
    
    cegnn = CEGNN(algebra=algebra, hidden_features=args.hidden_features, out_features=args.out_features, n_layers=args.n_layers, product_paths_sum=20, normalization=args.normalization)

    key = jax.random.PRNGKey(42 + measurement) 
    input = jax.random.normal(key, (args.batch_size * args.n_nodes, 3, 8))
    edges, edge_attr = get_edges_batch_jax(args.n_nodes, args.batch_size)
    edge_index = jnp.stack([_ for _ in edges], axis=0)
    edge_attr = jnp.broadcast_to(jnp.expand_dims(edge_attr, axis=-1), edge_attr.shape + (8,))

    params = cegnn.init(jax.random.PRNGKey(42 + measurement), input, edges, edge_attr)
    
    if measurement == 0:
        print(input.shape, edges[0].shape)
        print(f"Num params: {sum(x.size for x in jax.tree_leaves(params))}")
        
    jit_apply = jax.jit(cegnn.apply)

    # Warmup
    for _ in range(args.n_warmup):
        _ = jit_apply(params, input, edges, edge_attr)
    _.block_until_ready()
    
    # Benchmarking
    start = time.time()
    for _ in range(args.n_iter):
        _ = jit_apply(params, input, edges, edge_attr)
    _.block_until_ready()
    end = time.time()
    
    durations.append((end - start) / args.n_iter)

mean_duration = np.mean(durations)
std_duration = np.std(durations)

print(f'JAX, CEGNN, {args.n_nodes}: {mean_duration:.5f} +- {std_duration:.6f}s')
