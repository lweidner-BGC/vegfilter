import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from numba import njit, prange
import pgeof

# pgeof column indices (fixed layout, v0.2+)
_PGEOF_NAMES = [
    "linearity", "planarity", "scattering", "verticality",
    "normal_x", "normal_y", "normal_z",
    "length", "surface", "volume", "curvature",
]


# ---------------------------------------------------------------------------
# CSR helpers
# ---------------------------------------------------------------------------

def _radius_query_to_csr(ind) -> tuple:
    """Convert query_ball_point result (list of lists/arrays) to CSR.

    Returns (nn uint32, nn_ptr uint32) where nn_ptr has length M+1.
    Uses pre-allocated buffer + slice assignment to minimise GIL hold time.
    """
    lengths = np.fromiter((len(a) for a in ind), dtype=np.int64, count=len(ind))
    nn_ptr = np.zeros(len(ind) + 1, dtype=np.uint32)
    np.cumsum(lengths, out=nn_ptr[1:])
    total = int(nn_ptr[-1])
    if total == 0:
        return np.empty(0, dtype=np.uint32), nn_ptr
    nn = np.empty(total, dtype=np.uint32)
    for i, a in enumerate(ind):
        s = nn_ptr[i]
        nn[s:s + len(a)] = a   # numpy slice assignment — GIL-free at C level
    return nn, nn_ptr


# ---------------------------------------------------------------------------
# Numba neighborhood statistics
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _neighborhood_stats_csr(values, nn, nn_ptr):
    """Compute per-point mean and std of a scalar field over CSR neighborhoods."""
    n = len(nn_ptr) - 1
    means = np.full(n, np.nan, dtype=np.float32)
    stds = np.full(n, np.nan, dtype=np.float32)
    for i in prange(n):
        start = nn_ptr[i]
        end = nn_ptr[i + 1]
        if end <= start:
            continue
        v = values[nn[start:end]]
        m = np.float32(0.0)
        for x in v:
            m += x
        m /= len(v)
        means[i] = m
        var = np.float32(0.0)
        for x in v:
            d = x - m
            var += d * d
        stds[i] = np.sqrt(var / len(v))
    return means, stds


# ---------------------------------------------------------------------------
# Spatial subsampling
# ---------------------------------------------------------------------------

def spatial_subsample(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """Return indices of one representative point per voxel.

    Uses integer voxel keys; returns core_idx (int64, shape M).
    """
    keys = np.floor(xyz / voxel_size).astype(np.int64)
    # Cantor-like encoding for 3D keys using a large prime stride
    stride = np.array([1, 100003, 10000300009], dtype=np.int64)
    flat = keys @ stride
    _, first_occurrence = np.unique(flat, return_index=True)
    return first_occurrence.astype(np.int64)


# ---------------------------------------------------------------------------
# Main feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    xyz: np.ndarray,
    xyz_search: np.ndarray,
    radii: list,
    scalar_fields: dict,
    k_min: int = 3,
    query_batch: int = 300_000,
) -> pd.DataFrame:
    """Compute multi-scale geometric + scalar-field features.

    Parameters
    ----------
    xyz          : float32 (M, 3) — query/core points
    xyz_search   : float32 (N, 3) — search cloud (pass xyz if no core points)
    radii        : list of floats — neighborhood radii
    scalar_fields: {name: float32 (N,)} — fields defined on xyz_search
    k_min        : minimum neighbors for pgeof (points with fewer → NaN row)
    query_batch  : max query points per radius call (memory control)

    Returns
    -------
    pd.DataFrame with M rows and columns named
      "{feat}_r{radius:.4g}"  for pgeof features
      "{field}_mean_r{radius:.4g}" / "{field}_std_r{radius:.4g}" for scalar stats
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    xyz_search = np.asarray(xyz_search, dtype=np.float32)
    M = len(xyz)
    scalar_fields = {k: np.asarray(v, dtype=np.float32) for k, v in scalar_fields.items()}

    tree = cKDTree(xyz_search)

    # Build column names (deterministic — sorted radii)
    col_names = []
    for radius in sorted(radii):
        tag = f"r{radius:.4g}"
        col_names += [f"{n}_{tag}" for n in _PGEOF_NAMES]
        col_names += [f"{fn}_{stat}_{tag}" for fn in scalar_fields for stat in ("mean", "std")]

    result_arr = np.full((M, len(col_names)), np.nan, dtype=np.float32)
    n_batches = max(1, (M + query_batch - 1) // query_batch)

    for b_idx in range(n_batches):
        b_start = b_idx * query_batch
        b_end = min(b_start + query_batch, M)
        xyz_batch = xyz[b_start:b_end]
        B = b_end - b_start

        if n_batches > 1:
            print(f"      batch {b_idx+1}/{n_batches} ...", flush=True)

        col_offset = 0
        for radius in sorted(radii):
            ind = tree.query_ball_point(xyz_batch, r=radius)
            nn, nn_ptr = _radius_query_to_csr(np.array(ind, dtype=object))

            # --- pgeof geometric features ---
            counts = np.diff(nn_ptr.astype(np.int64))
            valid = counts >= k_min

            feat_block = np.full((B, len(_PGEOF_NAMES)), np.nan, dtype=np.float32)
            if valid.any():
                valid_idx = np.where(valid)[0]
                starts = nn_ptr[valid_idx].astype(np.int64)
                ends   = nn_ptr[valid_idx + 1].astype(np.int64)
                sub_lengths = ends - starts
                sub_ptr = np.zeros(len(valid_idx) + 1, dtype=np.uint32)
                np.cumsum(sub_lengths, out=sub_ptr[1:])
                total_sub = int(sub_ptr[-1])
                if total_sub > 0:
                    sub_nn = np.empty(total_sub, dtype=np.uint32)
                    for i, (s, e, ds) in enumerate(zip(starts, ends, sub_ptr)):
                        sub_nn[ds:ds + int(e - s)] = nn[s:e]
                    result = pgeof.compute_features(
                        xyz_search, sub_nn, sub_ptr, k_min=k_min,
                    )
                    feat_block[valid_idx] = result

            result_arr[b_start:b_end, col_offset:col_offset + len(_PGEOF_NAMES)] = feat_block
            col_offset += len(_PGEOF_NAMES)

            # --- scalar field neighborhood stats (reuse same nn/nn_ptr) ---
            for field_name, values in scalar_fields.items():
                means, stds = _neighborhood_stats_csr(values, nn, nn_ptr)
                result_arr[b_start:b_end, col_offset]     = means
                result_arr[b_start:b_end, col_offset + 1] = stds
                col_offset += 2

    return pd.DataFrame(result_arr, columns=col_names)


def _process_tile(xyz_tile, xyz_s, sf_tile, core_idx, radii, k_min, query_batch):
    """Process one tile — called from joblib threads."""
    df = extract_features(xyz_tile, xyz_s, radii, sf_tile, k_min=k_min,
                          query_batch=query_batch)
    return core_idx, df.values, list(df.columns)


def extract_features_tiled(
    xyz: np.ndarray,
    xyz_search: np.ndarray,
    radii: list,
    scalar_fields: dict,
    k_min: int = 3,
    tile_size: float = 20.0,
    n_jobs: int = 4,
    query_batch: int = 300_000,
) -> pd.DataFrame:
    """Like extract_features but processes XY tiles in parallel.

    Uses joblib loky processes for genuine multicore parallelism. Each worker
    imports its own copy of libraries (~1 GB RAM overhead per worker), so cap
    n_jobs based on available RAM (default 8 is safe on most machines).
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    xyz_search = np.asarray(xyz_search, dtype=np.float32)
    buf = max(radii)
    M = len(xyz)

    x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
    x_max, y_max = xyz[:, 0].max(), xyz[:, 1].max()

    x_tiles = np.arange(x_min, x_max + tile_size, tile_size)
    y_tiles = np.arange(y_min, y_max + tile_size, tile_size)

    # Build tile list (skip empty tiles)
    tiles = []
    for x0 in x_tiles:
        x1 = x0 + tile_size
        for y0 in y_tiles:
            y1 = y0 + tile_size
            core_mask = (
                (xyz[:, 0] >= x0) & (xyz[:, 0] < x1) &
                (xyz[:, 1] >= y0) & (xyz[:, 1] < y1)
            )
            if not core_mask.any():
                continue
            core_idx = np.where(core_mask)[0]
            xyz_tile = xyz[core_mask]
            search_mask = (
                (xyz_search[:, 0] >= x0 - buf) & (xyz_search[:, 0] < x1 + buf) &
                (xyz_search[:, 1] >= y0 - buf) & (xyz_search[:, 1] < y1 + buf)
            )
            xyz_s = xyz_search[search_mask]
            sf_tile = {k: v[search_mask] for k, v in scalar_fields.items()}
            tiles.append((xyz_tile, xyz_s, sf_tile, core_idx))

    n_tiles = len(tiles)
    print(f"    {n_tiles} non-empty tiles ...", flush=True)

    # Warm up numba JIT before processing tiles
    _dummy = np.zeros(2, dtype=np.float32)
    _neighborhood_stats_csr(_dummy, np.array([0, 1], dtype=np.uint32),
                            np.array([0, 1, 2], dtype=np.uint32))

    print(f"    dispatching {n_tiles} tiles ({n_jobs} workers) ...", flush=True)
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_process_tile)(xyz_t, xyz_s, sf_t, cidx, radii, k_min, query_batch)
        for xyz_t, xyz_s, sf_t, cidx in tiles
    )
    print(f"    assembling results ...", flush=True)

    result_arr = None
    col_names = None
    for core_idx, values, cols in results:
        if result_arr is None:
            col_names = cols
            result_arr = np.full((M, values.shape[1]), np.nan, dtype=np.float32)
        result_arr[core_idx] = values

    if result_arr is None:
        # Build empty frame with correct columns
        col_names = []
        for radius in sorted(radii):
            tag = f"r{radius:.4g}"
            col_names += [f"{n}_{tag}" for n in _PGEOF_NAMES]
            col_names += [f"{fn}_{stat}_{tag}"
                          for fn in scalar_fields for stat in ("mean", "std")]
        return pd.DataFrame(np.empty((M, len(col_names)), dtype=np.float32),
                            columns=col_names)

    return pd.DataFrame(result_arr, columns=col_names)
