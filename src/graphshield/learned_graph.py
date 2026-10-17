"""Historical graph inputs, biased-walk embeddings and inductive mean GraphSAGE.

Only observed records and candidate member IDs enter this module. CPU implementations
use PyTorch directly; no optional compiled graph extension is required.
"""

from collections import defaultdict
from itertools import combinations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CASE_FEATURES = ('member_count', 'internal_events', 'internal_directed_pairs', 'directed_density',
                 'repeated_event_fraction', 'event_support', 'internal_event_share', 'boundary_events',
                 'internal_amount_total', 'amount_cv', 'amount_similarity', 'shared_device_count',
                 'shared_device_member_fraction', 'max_repeated_in_degree', 'max_repeated_out_degree',
                 'internal_active_steps')
ACTIVITY_FEATURES = ('member_count', 'internal_events', 'boundary_events', 'internal_amount_total',
                     'amount_cv', 'internal_active_steps')
NODE_FEATURES = ('log_out_count', 'log_in_count', 'log_out_amount', 'log_in_amount',
                 'log_out_degree', 'log_in_degree', 'log_device_count', 'log_active_steps')
EMBEDDING_FEATURES = ('pair_cosine_mean', 'pair_cosine_std', 'pair_cosine_min',
                      'pair_cosine_max', 'missing_member_fraction')


def case_matrix(cases, features=CASE_FEATURES):
    return np.asarray([[c['features'][f] for f in features] for c in cases], dtype=np.float32).reshape(-1, len(features))


def historical_graph(transactions, observations, cutoff, window, device_limit=64):
    lower = max(1, cutoff - window)
    tx = sorted((r for r in transactions if lower <= r[4] < cutoff and r[5] < cutoff), key=lambda r: r[0])
    obs = sorted((r for r in observations if lower <= r[3] < cutoff and r[4] < cutoff), key=lambda r: r[0])
    nodes = sorted({a for r in tx for a in r[1:3]} | {r[1] for r in obs})
    index = {a: i for i, a in enumerate(nodes)}
    outgoing, incoming, devices, active = (defaultdict(set) for _ in range(4))
    shares = defaultdict(set)
    counts = np.zeros((len(nodes), 4), dtype=np.float64)
    edges = set()
    for _, u, v, amount, step, _ in tx:
        i, j = index[u], index[v]
        counts[i, 0] += 1
        counts[j, 1] += 1
        counts[i, 2] += float(amount)
        counts[j, 3] += float(amount)
        outgoing[u].add(v)
        incoming[v].add(u)
        active[u].add(step)
        active[v].add(step)
        if i != j:
            edges.update(((i, j), (j, i)))
    for _, account, device, step, _ in obs:
        devices[account].add(device)
        shares[device].add(index[account])
    skipped = 0
    for accounts in shares.values():
        if len(accounts) > device_limit:
            skipped += 1
            continue
        for i, j in combinations(sorted(accounts), 2):
            edges.update(((i, j), (j, i)))
    features = np.zeros((len(nodes), len(NODE_FEATURES)), dtype=np.float32)
    features[:, :4] = np.log1p(counts)
    for node, i in index.items():
        features[i, 4:] = np.log1p([len(outgoing[node]), len(incoming[node]), len(devices[node]), len(active[node])])
    edge_index = np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2).T.copy()
    return {'nodes': np.asarray(nodes, dtype=str), 'x': features, 'edges': edge_index,
            'oversized_devices_skipped': np.asarray(skipped, dtype=np.int64)}


def member_indices(cases, nodes):
    index = {str(a): i for i, a in enumerate(nodes)}
    width = max((len(c['members']) for c in cases), default=1)
    members = np.full((len(cases), width), -1, dtype=np.int64)
    for row, case in enumerate(cases):
        members[row, :len(case['members'])] = [index.get(a, -2) for a in case['members']]
    return members


def transition_weights(previous, neighbors, adjacency, p, q):
    return np.asarray([1 / p if v == previous else 1.0 if v in adjacency[previous] else 1 / q
                       for v in neighbors], dtype=np.float64)


def walk_pairs(edges, node_count, config, seed):
    rng = np.random.default_rng(seed)
    adjacent = [set() for _ in range(node_count)]
    for u, v in edges.T:
        adjacent[int(u)].add(int(v))
    neighbors = [np.asarray(sorted(v), dtype=np.int64) for v in adjacent]
    centers, contexts = [], []
    for _ in range(config['walks_per_node']):
        for start in rng.permutation(node_count):
            if not len(neighbors[start]):
                continue
            walk = [int(start)]
            while len(walk) < config['walk_length']:
                options = neighbors[walk[-1]]
                if not len(options):
                    break
                if len(walk) == 1:
                    nxt = rng.choice(options)
                else:
                    weights = transition_weights(walk[-2], options, adjacent, config['p'], config['q'])
                    nxt = rng.choice(options, p=weights / weights.sum())
                walk.append(int(nxt))
            for i, center in enumerate(walk):
                for j in range(max(0, i-config['context_size']), min(len(walk), i+config['context_size']+1)):
                    if i != j:
                        centers.append(center)
                        contexts.append(walk[j])
    return np.asarray(centers, dtype=np.int64), np.asarray(contexts, dtype=np.int64)


def node2vec(edges, node_count, config, seed):
    """Second-order p/q walks, symmetric windows, skip-gram negative sampling.

    The input embedding table is retained; isolated nodes have an explicit zero vector.
    Embeddings are fitted independently using only the current snapshot's edges.
    """
    torch.manual_seed(seed)
    centers, contexts = walk_pairs(edges, node_count, config, seed)
    if not len(centers):
        return np.zeros((node_count, config['dimensions']), np.float32), {'pairs': 0, 'loss': []}
    inputs = nn.Embedding(node_count, config['dimensions'])
    outputs = nn.Embedding(node_count, config['dimensions'])
    nn.init.uniform_(inputs.weight, -.5 / config['dimensions'], .5 / config['dimensions'])
    nn.init.zeros_(outputs.weight)
    optimizer = torch.optim.Adam(list(inputs.parameters()) + list(outputs.parameters()), lr=config['learning_rate'])
    center_ids, context_ids = torch.from_numpy(centers), torch.from_numpy(contexts)
    frequency = np.bincount(contexts, minlength=node_count).astype(np.float64) ** .75
    noise = torch.tensor(frequency / frequency.sum(), dtype=torch.float32)
    history = []
    for _ in range(config['epochs']):
        permutation = torch.randperm(len(centers))
        total = 0.0
        for batch in permutation.split(config['batch_size']):
            source, target = inputs(center_ids[batch]), outputs(context_ids[batch])
            negative_ids = torch.multinomial(noise, len(batch) * config['negative_samples'], replacement=True)
            negative = outputs(negative_ids.reshape(len(batch), config['negative_samples']))
            loss = (-F.logsigmoid((source * target).sum(1))
                    - F.logsigmoid(-(source[:, None, :] * negative).sum(2)).sum(1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        history.append(total / len(centers))
    result = inputs.weight.detach().numpy().copy()
    degree = np.bincount(edges[0], minlength=node_count)
    result[degree == 0] = 0
    return result, {'pairs': len(centers), 'loss': history}


def embedding_features(embeddings, members):
    """Within-case cosines avoid sharing raw, unaligned coordinates across worlds."""
    rows = []
    for row in members:
        present = row[row >= 0]
        vectors = embeddings[present]
        norms = np.linalg.norm(vectors, axis=1)
        vectors = vectors[norms > 1e-10] / norms[norms > 1e-10, None]
        # Padding is not a missing member; count actual membership via caller's width mask.
        values = (vectors @ vectors.T)[np.triu_indices(len(vectors), k=1)]
        # Unknown entries use -2; -1 is padding.
        count = int(np.count_nonzero(row != -1))
        missing = 1 - len(vectors) / count if count else 1.0
        rows.append([float(values.mean()), float(values.std()), float(values.min()), float(values.max()), missing]
                    if len(values) else [0.0, 0.0, 0.0, 0.0, missing])
    return np.asarray(rows, np.float32).reshape(-1, len(EMBEDDING_FEATURES))


class MeanLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim * 2, output_dim)

    def forward(self, x, edges, use_edges=True):
        neighbor = torch.zeros_like(x)
        if use_edges and edges.numel():
            source, destination = edges
            neighbor.index_add_(0, destination, x[source])
            degree = torch.bincount(destination, minlength=len(x)).clamp_min(1).to(x.dtype)
            neighbor = neighbor / degree[:, None]
        return F.normalize(F.relu(self.linear(torch.cat((x, neighbor), dim=1))), dim=1)


class CaseNetwork(nn.Module):
    def __init__(self, hidden, use_edges=True):
        super().__init__()
        self.use_edges = use_edges
        self.first = MeanLayer(len(NODE_FEATURES), hidden)
        self.second = MeanLayer(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(2 * hidden + len(CASE_FEATURES), hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x, edges, members, case_features):
        x = self.second(self.first(x, edges, self.use_edges), edges, self.use_edges)
        valid = members >= 0
        # Graphs with cases always contain at least one member; unknowns are masked.
        selected = x[members.clamp_min(0)]
        mean = (selected * valid[:, :, None]).sum(1) / valid.sum(1).clamp_min(1)[:, None]
        maximum = selected.masked_fill(~valid[:, :, None], -torch.inf).amax(1)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return self.head(torch.cat((mean, maximum, case_features), 1)).squeeze(1)
