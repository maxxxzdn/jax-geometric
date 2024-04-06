import argparse
import tqdm
import torch
import numpy as np

from models.cegnn.torch.cegnn import CEGNN
from benchmarks.utils import get_edges_batch_torch

assert torch.cuda.is_available()


parser = argparse.ArgumentParser()
# model hyperparameters
parser.add_argument('--hidden_features', type=int, default=32)
parser.add_argument('--edge_features', type=int, default=1)
parser.add_argument('--out_features', type=int, default=1)
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--normalization', type=int, default=1)
# input parameters
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_nodes', type=int, default=5)
# benchmarking parameters
parser.add_argument('--n_warmup', type=int, default=10)
parser.add_argument('--n_iter', type=int, default=100)
parser.add_argument('--n_measurements', type=int, default=10)
args = parser.parse_args()


durations = []

for measurement in tqdm.tqdm(range(args.n_measurements)):
    
    cegnn = CEGNN(3, args.hidden_features, args.out_features, args.edge_features, args.n_layers).cuda()
    
    input = torch.randn((args.batch_size * args.n_nodes, 3, 8), dtype=torch.float).cuda()
    edges, edge_attr = get_edges_batch_torch(args.n_nodes, args.batch_size)
    edges = [edge.cuda() for edge in edges]
    edge_attr = edge_attr.unsqueeze(-1).expand(-1, -1, 8).cuda()

    if measurement == 0:
        print(input.shape, edges[0].shape)
        print('Num. params:', sum(p.numel() for p in cegnn.parameters() if p.requires_grad))
        
    # Warmup
    for _ in range(args.n_warmup):
        _ = cegnn(input, edges, edge_attr)
    torch.cuda.synchronize()

    # Benchmarking
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.n_iter):
        _ = cegnn(input, edges, edge_attr)
    end.record()
    torch.cuda.synchronize()
    
    durations.append(0.001 * start.elapsed_time(end) / args.n_iter) # average in seconds

mean_duration = np.mean(durations)
std_duration = np.std(durations)

print(f'PyTorch, CEGNN, {args.n_nodes}: {mean_duration:.5f} +- {std_duration:.6f}s')
