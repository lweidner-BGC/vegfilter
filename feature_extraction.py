import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from numba import njit, prange
from joblib import Parallel, delayed
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

def _radius_query_to_csr(ind: np.ndarray) -> tuple:
    """Convert query_radius result (object array of int64 arrays) to CSR.

    Returns (nn uint32, nn_ptr uint32) where nn_ptr has length M+1.
    """
    lengths = np.array([len(a) for a in ind], dtype=np.int64)
    nn_ptr = np.zeros(len(ind) + 1, dtype=np.uint32)
    np.cumsum(lengths, out=nn_ptr[1:])
    if nn_ptr[-1] > 0:
        nn = np.concatenate([np.asarray(a, dtype=np.uint32) for a in ind])
    else:
        nn = np.empty(0, dtype=np.uint32)
    return nn, nn_ptr


# ---------------------------------------------------------------------------
# Numba neighborhood statistics
# ---------------------------------------------------------------------------

@njit(parallel=True)
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
) -> pd.DataFrame:
    """Compute multi-scale geometric + scalar-field features.

    Parameters
    ----------
    xyz          : float32 (M, 3) — query/core points
    xyz_search   : float32 (N, 3) — search cloud (pass xyz if no core points)
    radii        : list of floats — neighborhood radii
    scalar_fields: {name: float32 (N,)} — fields defined on xyz_search
    k_min        : minimum neighbors for pgeof (points with fewer → NaN row)

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

    tree = None  # lazy; rebuilt if xyz_search changes between radii (it won't here)
    tree = KDTree(xyz_search)

    all_cols = {}

    for radius in sorted(radii):
        tag = f"r{radius:.4g}"

        ind = tree.query_ball_point(xyz, r=radius, workers=1)
        nn, nn_ptr = _radius_query_to_csr(np.array(ind, dtype=object))

        # --- pgeof geometric features ---
        # Mask points with enough neighbors
        counts = np.diff(nn_ptr.astype(np.int64))
        valid = counts >= k_min

        feat_block = np.full((M, len(_PGEOF_NAMES)), np.nan, dtype=np.float32)
        if valid.any():
            # Build sub-CSR for valid points only
            valid_idx = np.where(valid)[0]
            sub_nn_list = [nn[nn_ptr[i]:nn_ptr[i + 1]] for i in valid_idx]
            sub_lengths = np.array([len(a) for a in sub_nn_list], dtype=np.int64)
            sub_ptr = np.zeros(len(valid_idx) + 1, dtype=np.uint32)
            np.cumsum(sub_lengths, out=sub_ptr[1:])
            sub_nn = np.concatenate(sub_nn_list) if sub_ptr[-1] > 0 \
                else np.empty(0, dtype=np.uint32)

            result = pgeof.compute_features(
                xyz_search,
                sub_nn,
                sub_ptr,
                k_min=k_min,
            )
            feat_block[valid_idx] = result

        for j, name in enumerate(_PGEOF_NAMES):
            all_cols[f"{name}_{tag}"] = feat_block[:, j]

        # --- scalar field neighborhood stats (reuse same nn/nn_ptr) ---
        for field_name, values in scalar_fields.items():
            means, stds = _neighborhood_stats_csr(values, nn, nn_ptr)
            all_cols[f"{field_name}_mean_{tag}"] = means
            all_cols[f"{field_name}_std_{tag}"] = stds

    return pd.DataFrame(all_cols)


def _process_tile(xyz_tile, xyz_s, sf_tile, core_idx, radii, k_min):
    """Process one tile — runs in a worker thread."""
    df = extract_features(xyz_tile, xyz_s, radii, sf_tile, k_min=k_min)
    return core_idx, df.values


def extract_features_tiled(
    xyz: np.ndarray,
    xyz_search: np.ndarray,
    radii: list,
    scalar_fields: dict,
    k_min: int = 3,
    tile_size: float = 20.0,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Like extract_features but processes overlapping XY tiles in parallel.

    Tiles are independent so all CPU cores are used via joblib threads.
    workers=1 is used inside each tile to avoid nested thread contention.
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
    print(f"    {n_tiles} non-empty tiles, running on {n_jobs} workers ...", flush=True)

    results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
        delayed(_process_tile)(xyz_t, xyz_s, sf_t, cidx, radii, k_min)
        for xyz_t, xyz_s, sf_t, cidx in tiles
    )

    # Assemble results in original row order
    col_names = None
    result_arr = np.full((M, 1), np.nan, dtype=np.float32)  # placeholder shape
    for core_idx, values in results:
        if col_names is None:
            col_names = values  # will be replaced below
            result_arr = np.full((M, values.shape[1]), np.nan, dtype=np.float32)
        result_arr[core_idx] = values

    # Recover column names from first result's DataFrame shape
    # (re-derive from a dummy single-point call if needed — simpler: use last df)
    # Column names are deterministic from radii+scalar_fields, so rebuild them
    dummy_cols = []
    for radius in sorted(radii):
        tag = f"r{radius:.4g}"
        dummy_cols += [f"{n}_{tag}" for n in _PGEOF_NAMES]
        dummy_cols += [f"{fn}_{stat}_{tag}"
                       for fn in scalar_fields for stat in ("mean", "std")]

    return pd.DataFrame(result_arr, columns=dummy_cols)
