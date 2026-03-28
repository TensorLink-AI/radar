"""Frozen data preparation for graph complexity task.

Generates random graphs, samples biased walks, projects into modalities,
discretises into tokens, and computes complexity profile. All randomness
is deterministic from SEED env var.

Exports: CONTEXT_LEN, PREDICTION_LEN, VOCAB_SIZE, MODALITY,
         PREDICTION_MODE, MARGINAL_ENTROPY, BIN_CENTRES, get_dataloader(),
         validate()
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import torch

# ── Task constants ───────────────────────────────────────────────
CONTEXT_LEN = 256
PREDICTION_LEN = 64
VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", "1024"))
MODALITY = os.environ.get("MODALITY", "tokens")
PREDICTION_MODE = os.environ.get("PREDICTION_MODE", "direct")
SEED = int(os.environ.get("SEED", "42"))
EVAL_SPLIT_SEED = int(os.environ.get("EVAL_SPLIT_SEED", "7"))

# Graph parameters from challenge
GRAPH_TYPE = os.environ.get("GRAPH_TYPE", "er")
GRAPH_NODES = int(os.environ.get("GRAPH_NODES", "500"))
GRAPH_EDGES = int(os.environ.get("GRAPH_EDGES", "5000"))
KAPPA = float(os.environ.get("KAPPA", "1.0"))

N_WALKS = int(os.environ.get("N_WALKS", "2000"))
WALK_LEN = CONTEXT_LEN + PREDICTION_LEN + 10  # extra margin

# Lazily computed globals
_BIN_EDGES: np.ndarray | None = None
_BIN_CENTRES: np.ndarray | None = None
_MARGINAL_ENTROPY: float | None = None
_TRAIN_DATA: dict | None = None
_VAL_DATA: dict | None = None

BIN_CENTRES: np.ndarray = np.array([])
MARGINAL_ENTROPY: float = 0.0


# ── Graph generation ─────────────────────────────────────────────

def generate_graph(
    graph_type: str, n_nodes: int, n_edges: int, seed: int,
) -> np.ndarray:
    """Generate a random graph as a symmetric adjacency matrix.

    Args:
        graph_type: "er" (Erdos-Renyi) or "ba" (Barabasi-Albert)
        n_nodes: number of nodes
        n_edges: target total edges
        seed: random seed

    Returns:
        (n_nodes, n_nodes) symmetric binary adjacency matrix.
    """
    rng = np.random.RandomState(seed)

    if graph_type == "ba":
        m = max(1, n_edges // n_nodes)
        adj = _barabasi_albert(n_nodes, m, rng)
    else:
        p = min(1.0, 2.0 * n_edges / (n_nodes * (n_nodes - 1)))
        adj = _erdos_renyi(n_nodes, p, rng)

    return adj


def _erdos_renyi(n: int, p: float, rng: np.random.RandomState) -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
    # Ensure no isolated nodes
    for i in range(n):
        if adj[i].sum() == 0:
            j = rng.randint(0, n - 1)
            if j >= i:
                j = (j + 1) % n
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    return adj


def _barabasi_albert(n: int, m: int, rng: np.random.RandomState) -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.float32)
    # Start with a complete graph on m+1 nodes
    init_nodes = min(m + 1, n)
    for i in range(init_nodes):
        for j in range(i + 1, init_nodes):
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    degrees = adj.sum(axis=1)
    for new_node in range(init_nodes, n):
        total_deg = degrees[:new_node].sum()
        if total_deg < 1e-8:
            probs = np.ones(new_node) / new_node
        else:
            probs = degrees[:new_node] / total_deg
        targets = rng.choice(new_node, size=min(m, new_node), replace=False, p=probs)
        for t in targets:
            adj[new_node, t] = 1.0
            adj[t, new_node] = 1.0
            degrees[t] += 1
        degrees[new_node] = len(targets)

    return adj


# ── Walk sampling ────────────────────────────────────────────────

def graph_walks(
    adj: np.ndarray, kappa: float,
    n_walks: int, walk_len: int, seed: int,
) -> np.ndarray:
    """Sample biased random walks on a graph.

    kappa controls bias: 0 = uniform (max entropy),
    higher = prefer high-degree neighbours (more predictable).

    Returns:
        (n_walks, walk_len) integer node IDs.
    """
    rng = np.random.RandomState(seed)
    n_nodes = adj.shape[0]
    degrees = adj.sum(axis=1)

    walks = np.zeros((n_walks, walk_len), dtype=np.int64)
    for w in range(n_walks):
        node = rng.randint(0, n_nodes)
        walks[w, 0] = node
        for t in range(1, walk_len):
            neighbours = np.where(adj[node] > 0)[0]
            if len(neighbours) == 0:
                walks[w, t] = node
                continue
            if kappa < 1e-8:
                probs = np.ones(len(neighbours)) / len(neighbours)
            else:
                weights = degrees[neighbours] ** kappa
                probs = weights / weights.sum()
            node = rng.choice(neighbours, p=probs)
            walks[w, t] = node
    return walks


# ── Modality projection ─────────────────────────────────────────

class ModalityProjection:
    """Project node ID walks into different signal modalities."""

    def __init__(self, adj: np.ndarray, modality: str, seed: int):
        self.modality = modality
        self.n_nodes = adj.shape[0]
        rng = np.random.RandomState(seed)
        degrees = adj.sum(axis=1)

        if modality == "tokens":
            # Zipf-shuffled permutation of node IDs
            self.perm = rng.permutation(self.n_nodes)
        elif modality == "continuous":
            # Spectral embedding from graph Laplacian (eigenvectors 1-3)
            self.embedding = _spectral_embedding(adj, n_dims=3, seed=seed)
        elif modality == "waveform":
            # Sinusoidal snippet per node
            max_deg = max(degrees.max(), 1.0)
            self.frequencies = np.zeros(self.n_nodes)
            self.amplitudes = np.zeros(self.n_nodes)
            eigvals = _spectral_positions(adj, seed=seed)
            for i in range(self.n_nodes):
                self.frequencies[i] = 0.5 + 2.0 * eigvals[i]
                self.amplitudes[i] = math.log1p(degrees[i]) / math.log1p(max_deg)
        elif modality == "rms_energy":
            max_deg = max(degrees.max(), 1.0)
            self.energy = np.array([
                math.log1p(degrees[i]) / math.log1p(max_deg)
                for i in range(self.n_nodes)
            ])

    def project(self, walks: np.ndarray) -> np.ndarray:
        """Project integer node-ID walks to float sequences.

        Args:
            walks: (n_walks, walk_len) node IDs

        Returns:
            (n_walks, walk_len) float values for discretisation.
        """
        if self.modality == "tokens":
            return self.perm[walks].astype(np.float64)
        elif self.modality == "continuous":
            # Use first eigenvector for 1D projection
            return self.embedding[walks, 0]
        elif self.modality == "waveform":
            n_walks, walk_len = walks.shape
            result = np.zeros((n_walks, walk_len), dtype=np.float64)
            t = np.arange(walk_len, dtype=np.float64)
            for w in range(n_walks):
                for s in range(walk_len):
                    node = walks[w, s]
                    result[w, s] = (
                        self.amplitudes[node]
                        * math.sin(2.0 * math.pi * self.frequencies[node] * s / walk_len)
                    )
            return result
        elif self.modality == "rms_energy":
            return self.energy[walks]
        raise ValueError(f"Unknown modality: {self.modality}")


def _spectral_embedding(
    adj: np.ndarray, n_dims: int = 3, seed: int = 42,
) -> np.ndarray:
    """Graph Laplacian spectral embedding."""
    degrees = adj.sum(axis=1)
    D = np.diag(degrees)
    L = D - adj
    # Normalised Laplacian for numerical stability
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-8)))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    try:
        from scipy.linalg import eigh
        # Get smallest eigenvectors (skip the trivial constant one)
        n = min(n_dims + 1, adj.shape[0])
        eigenvalues, eigenvectors = eigh(L_norm, subset_by_index=[0, n - 1])
        return eigenvectors[:, 1:n_dims + 1]  # skip Fiedler eigenvector 0
    except Exception:
        rng = np.random.RandomState(seed)
        return rng.randn(adj.shape[0], n_dims) * 0.1


def _spectral_positions(adj: np.ndarray, seed: int = 42) -> np.ndarray:
    """Normalised spectral position for each node (0-1 scale)."""
    embedding = _spectral_embedding(adj, n_dims=1, seed=seed)
    vals = embedding[:, 0]
    vmin, vmax = vals.min(), vals.max()
    if vmax - vmin < 1e-8:
        return np.full(len(vals), 0.5)
    return (vals - vmin) / (vmax - vmin)


# ── Discretisation ───────────────────────────────────────────────

def discretise(
    values: np.ndarray, vocab_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentile-based binning into vocab_size bins.

    Returns:
        (tokens, bin_edges, bin_centres) where tokens has same shape as values.
    """
    flat = values.ravel()
    # Percentile-based bin edges for roughly uniform marginals
    percentiles = np.linspace(0, 100, vocab_size + 1)
    bin_edges = np.percentile(flat, percentiles)
    # Ensure strictly increasing edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        bin_edges = np.array([flat.min() - 1, flat.max() + 1])

    actual_vocab = len(bin_edges) - 1
    tokens = np.clip(np.digitize(flat, bin_edges[1:-1]), 0, actual_vocab - 1)
    tokens = tokens.reshape(values.shape)

    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # Pad or truncate to exact vocab_size
    if len(bin_centres) < vocab_size:
        pad = np.full(vocab_size - len(bin_centres), bin_centres[-1])
        bin_centres = np.concatenate([bin_centres, pad])
        bin_edges_new = np.linspace(bin_edges[0], bin_edges[-1], vocab_size + 1)
        bin_edges = bin_edges_new
    elif len(bin_centres) > vocab_size:
        bin_centres = bin_centres[:vocab_size]
        bin_edges = bin_edges[:vocab_size + 1]
        tokens = np.clip(tokens, 0, vocab_size - 1)

    return tokens.astype(np.int64), bin_edges, bin_centres


# ── Data preparation ─────────────────────────────────────────────

def _prepare_data() -> tuple[dict, dict]:
    """Generate data, project, discretise, compute complexity. Cached."""
    global _TRAIN_DATA, _VAL_DATA, _BIN_EDGES, _BIN_CENTRES, _MARGINAL_ENTROPY
    global BIN_CENTRES, MARGINAL_ENTROPY

    if _TRAIN_DATA is not None and _VAL_DATA is not None:
        return _TRAIN_DATA, _VAL_DATA

    # 1. Generate graph
    adj = generate_graph(GRAPH_TYPE, GRAPH_NODES, GRAPH_EDGES, seed=SEED)

    # 2. Sample walks
    walks = graph_walks(adj, KAPPA, N_WALKS, WALK_LEN, seed=SEED + 1)

    # 3. Project to modality
    proj = ModalityProjection(adj, MODALITY, seed=SEED + 2)
    projected = proj.project(walks)

    # 4. Per-window z-score normalisation
    from complexity import normalise_windows
    projected = normalise_windows(projected, window_size=CONTEXT_LEN)

    # 5. Discretise
    tokens, bin_edges, bin_centres = discretise(projected, VOCAB_SIZE)
    _BIN_EDGES = bin_edges
    _BIN_CENTRES = bin_centres
    BIN_CENTRES = bin_centres

    # 6. Compute marginal entropy
    from complexity import marginal_entropy as compute_marginal
    _MARGINAL_ENTROPY = compute_marginal(tokens)
    MARGINAL_ENTROPY = _MARGINAL_ENTROPY

    # 7. Compute and cache complexity profile
    try:
        from complexity import compute_complexity_profile
        profile = compute_complexity_profile(tokens, projected)
        profile_path = os.environ.get(
            "COMPLEXITY_PROFILE_PATH", "/workspace/complexity_profile.json",
        )
        os.makedirs(os.path.dirname(profile_path) or ".", exist_ok=True)
        with open(profile_path, "w") as f:
            json.dump(profile, f, default=_json_default)
    except Exception:
        pass  # Non-fatal — complexity profile is informational

    # 8. Train/val split on WALKS (not windows) to prevent data leakage
    split_rng = np.random.RandomState(EVAL_SPLIT_SEED)
    indices = split_rng.permutation(N_WALKS)
    val_size = max(1, N_WALKS // 5)  # 20% validation
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    seq_len = CONTEXT_LEN + PREDICTION_LEN

    def _make_windows(walk_tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Slice walks into (context, target) windows."""
        n_walks, wl = walk_tokens.shape
        x_list, y_list = [], []
        for w in range(n_walks):
            for start in range(0, wl - seq_len + 1, PREDICTION_LEN):
                end = start + seq_len
                if end > wl:
                    break
                x_list.append(walk_tokens[w, start:start + CONTEXT_LEN])
                y_list.append(walk_tokens[w, start + CONTEXT_LEN:end])
        if not x_list:
            # Fallback: at least one window per walk
            for w in range(n_walks):
                x_list.append(walk_tokens[w, :CONTEXT_LEN])
                y_list.append(walk_tokens[w, CONTEXT_LEN:CONTEXT_LEN + PREDICTION_LEN])
        return np.stack(x_list), np.stack(y_list)

    train_x, train_y = _make_windows(tokens[train_idx])
    val_x, val_y = _make_windows(tokens[val_idx])

    _TRAIN_DATA = {"x": train_x, "y": train_y}
    _VAL_DATA = {"x": val_x, "y": val_y}

    return _TRAIN_DATA, _VAL_DATA


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


# ── Public API ───────────────────────────────────────────────────

def get_dataloader(batch_size: int = 64, seed: int = 42):
    """Yield training batches of (x_tokens, y_tokens) as LongTensors."""
    train, _ = _prepare_data()
    x_all, y_all = train["x"], train["y"]
    n = len(x_all)
    rng = np.random.RandomState(seed)

    while True:
        indices = rng.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            yield {
                "x_tokens": torch.from_numpy(x_all[idx]).long(),
                "y_tokens": torch.from_numpy(y_all[idx]).long(),
            }


def validate(
    model, n_batches: int = 10, batch_size: int = 32, seed: int = 42,
) -> dict:
    """Evaluate model on validation set. Always teacher-forced."""
    _, val = _prepare_data()
    x_all, y_all = val["x"], val["y"]
    n = len(x_all)

    device = next(model.parameters()).device
    total_ce = 0.0
    total_tokens = 0
    n_eval = min(n_batches, max(1, n // batch_size))

    with torch.no_grad():
        for i in range(n_eval):
            start = i * batch_size
            end = min(start + batch_size, n)
            if start >= n:
                break

            x = torch.from_numpy(x_all[start:end]).long().to(device)
            y = torch.from_numpy(y_all[start:end]).long().to(device)

            # Always teacher-forced for evaluation
            inp = torch.cat([x, y[:, :-1]], dim=1)
            logits = model(inp)
            # Extract last prediction_len positions
            logits_pred = logits[:, -PREDICTION_LEN:]
            ce = torch.nn.functional.cross_entropy(
                logits_pred.reshape(-1, logits_pred.size(-1)),
                y.reshape(-1),
                reduction="sum",
            )
            total_ce += ce.item()
            total_tokens += y.numel()

    raw_ce = total_ce / max(total_tokens, 1)
    h_marginal = MARGINAL_ENTROPY if MARGINAL_ENTROPY > 0 else 1.0
    normalised = raw_ce / h_marginal

    return {
        "raw_ce": raw_ce,
        "normalised_ce": normalised,
        "universal_ce": raw_ce / max(math.log(VOCAB_SIZE), 1e-8),
    }
