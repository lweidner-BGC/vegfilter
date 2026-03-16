# Fast eigenfeature extraction from LiDAR point clouds

**pgeof is the best-in-class open-source solution** for computing the full Weinmann et al. (2015) eigenfeature set—linearity, planarity, omnivariance, anisotropy, surface variation, sphericity, verticality, eigenentropy, and optimal neighborhood size—in a single pass over point neighborhoods with parallel C++ performance and clean Python bindings. PDAL's `filters.covariancefeatures` is the strongest pipeline-oriented alternative, computing all features in one KD-tree pass with multithreading. For GPU acceleration, no turnkey solution exists, but **PyTorch3D's `knn_points` combined with `torch.linalg.eigh`** delivers a fully GPU-accelerated eigenfeature pipeline in roughly 20 lines of code. The deep-learning preprocessing pipelines (RandLA-Net, KPConv) do not compute classical eigenfeatures and are not useful for this purpose.

---

## pgeof delivers the complete Weinmann framework in a single call

The **pgeof** library (Point Geometric Features, `github.com/drprojects/point_geometric_features`) is the most targeted solution for this problem. Developed by Damien Robert at IGN France / LASTIG—extracted and refined from the Superpoint Graph codebase (Landrieu & Simonovsky, CVPR 2018)—it computes **11 standard eigenfeatures** in a single neighborhood query: linearity, planarity, scattering, omnivariance, anisotropy, eigenentropy, sum of eigenvalues, surface variation, verticality, and normal vectors.

Three computation modes cover the key use cases. `pgeof.compute_features()` handles fixed-*k* neighborhoods. `pgeof.compute_features_multiscale()` evaluates features across an array of neighborhood sizes. Most critically, `pgeof.compute_features_optimal()` **implements Weinmann et al.'s eigenentropy-based optimal neighborhood size selection**, returning 12 values per point (11 features + optimal *k*). This is the only library found that implements the full Weinmann (2015) framework out of the box.

The architecture directly addresses the user's pain points. Neighborhoods are stored in **CSR (Compressed Sparse Row) format**, allowing variable-size neighborhoods and flexible query-point specification—effectively enabling core-point subsampling by constructing CSR indices only for a subset of points while searching against the full cloud. The C++ core uses **nanoflann** for KD-tree search (thread-safe queries), **Eigen** for 3×3 eigendecomposition, and **Taskflow** for task-parallel dispatch across all CPU cores. Python bindings via **nanobind** are lightweight and pip-installable via PEP 517 / scikit-build-core. The library is MIT-licensed, actively maintained (~99 commits), and the direct ancestor of feature computation in `drprojects/superpoint_transformer` (ICCV 2023).

| Property | pgeof |
|---|---|
| Features | 11 standard + optimal *k* |
| Single-pass | ✅ One KD-tree build, one neighbor query |
| Core points | ✅ Via CSR neighborhood specification |
| Threading | Taskflow (C++ task parallelism) |
| Python bindings | nanobind (high quality) |
| KNN backend | nanoflann |
| Install | `pip install` (build from source) |

---

## PDAL's covariancefeatures filter is the pipeline workhorse

**PDAL** (`github.com/PDAL/PDAL`, v2.10.0, February 2026) provides the most integrated solution for batch/pipeline eigenfeature computation through its `filters.covariancefeatures` filter. This single filter computes **all standard eigenfeatures in one pass**—linearity, planarity, scattering, omnivariance, anisotropy, eigenentropy, eigenvalue sum, surface variation, verticality (two formulations), and density—from a single KD-tree construction and neighbor search. The `feature_set` parameter accepts `"all"` or a comma-separated list of specific features. An internal `threads` parameter (default 1) enables **OpenMP-style parallelism** within the filter.

PDAL also provides `filters.optimalneighborhood`, which finds the *k* minimizing eigenentropy (Weinmann's criterion), outputting `OptimalKNN` and `OptimalRadius` dimensions that feed directly into `covariancefeatures` via the `optimized: true` flag. This two-filter chain (`optimalneighborhood` → `covariancefeatures`) replicates the Weinmann pipeline, though each filter builds its own KD-tree.

The **Python bindings** (`pip install pdal`, v3.5.3) are well-maintained, offering pipeline construction from JSON or programmatic `Stage` chaining with full NumPy integration:

```python
import pdal
pipeline = pdal.Reader("input.las") | pdal.Filter.covariancefeatures(
    knn=20, threads=8, feature_set="all"
)
pipeline.execute()
features = pipeline.arrays[0]  # structured numpy array
```

The primary limitation is **no native core-point support**. PDAL lacks PCL's `setSearchSurface()` concept. The `where` clause can restrict which points get processed, and pre-filtering with `filters.sample` or voxel downsampling is a workaround, but the search neighborhood always comes from the same point set. Each filter in a pipeline also builds its own KD-tree independently, so chaining `eigenvalues` → `covariancefeatures` → `normal` triples the tree construction cost. The solution is to use `covariancefeatures` with `feature_set: "all"` as a single filter.

---

## PCL offers core-point support but requires custom eigenfeature code

**PCL** (`github.com/PointCloudLibrary/pcl`, v1.15.1) is the only major library with first-class **core-point subsampling** via the `setInputCloud()` / `setSearchSurface()` pattern. You can compute features on a downsampled cloud while searching neighbors in the full cloud—exactly the workflow the user wants. A pre-built `pcl::search::KdTree` can be **shared across estimators** via `setSearchMethod()`, avoiding redundant tree construction.

However, PCL does **not provide a built-in eigenfeature class**. There is no equivalent of PDAL's `covariancefeatures`. Computing planarity, linearity, or omnivariance requires manually iterating over points, calling `computeMeanAndCovarianceMatrix()` + `pcl::eigen33()`, and deriving features from eigenvalues. The OMP-parallelized estimators (`NormalEstimationOMP`, `FPFHEstimationOMP`, `PrincipalCurvaturesEstimation` as of 1.15.0) each perform independent neighbor queries—**cached neighbor lists are not shared** between estimators, so computing normals + curvatures + custom eigenfeatures means 3+ separate neighbor search passes even when sharing the same KD-tree.

PCL 1.15.1 now supports **nanoflann as a KD-tree backend** (faster than the legacy FLANN dependency). Threading uses **OpenMP or TBB** depending on build configuration. The critical weakness is Python bindings: **python-pcl is archived**, **pclpy is effectively unmaintained** (no PyPI releases in 12+ months, Windows-only, Python 3.6). For a Python-centric pipeline, PCL is impractical without significant wrapping effort.

---

## jakteristics provides the simplest Python eigenfeature tool

**jakteristics** (`github.com/jakarto3d/jakteristics`) is a Cython library computing **13 eigenfeatures** (eigenvalue sum, omnivariance, eigenentropy, anisotropy, planarity, linearity, PCA1, PCA2, surface variation, sphericity, verticality, normal x/y/z) with **OpenMP multithreading** and direct LAS file I/O. The API is minimal:

```python
from jakteristics import compute_features
features = compute_features(xyz, search_radius=0.15, feature_names=["planarity", "linearity"])
```

A CLI tool (`jakteristics input.las output.las --search-radius 0.15 --num-threads 4`) makes batch processing trivial. It claims **≥2× faster than CloudCompare** for equivalent computation. The main limitations: **radius-based search only** (no k-NN or optimal neighborhood selection), and **no core-point support**—features are computed for all input points. For fixed-radius workflows where simplicity matters, jakteristics is excellent. For Weinmann-style optimal neighborhoods, use pgeof instead.

---

## Open3D exposes covariances but lacks core-point separation

**Open3D** (v0.19.0, `github.com/isl-org/Open3D`) provides `compute_point_cloud_covariance()`, which returns per-point **3×3 covariance matrices** from KD-tree neighborhood search. Combining this with `np.linalg.eigvalsh()` yields eigenvalues from which all features can be derived in a few lines of vectorized NumPy. Python bindings are **first-class** (pybind11, pip-installable, extensive documentation).

The key gap is **no search-surface / core-point distinction**. `estimate_normals()` and `compute_point_cloud_covariance()` operate on the cloud itself with no way to specify separate query points and search points. To achieve core-point behavior, you must either compute covariances for all points and index afterward (wasteful), or manually build a KD-tree and implement per-query-point search (losing the C++ optimization). Threading uses a mix of **OpenMP and TBB** internally, though some users report inconsistent parallel utilization. The tensor API (`open3d.t.geometry`) supports GPU hybrid search for normal estimation, but **GPU covariance/eigenvalue computation is not exposed**—it remains CPU-only.

---

## GPU acceleration requires assembling primitives, not a turnkey library

No existing library provides a complete GPU-accelerated eigenfeature pipeline. The most practical approach combines **PyTorch3D** (`facebookresearch/pytorch3d`) for GPU KNN with PyTorch's batched eigendecomposition:

```python
from pytorch3d.ops import knn_points
import torch

# GPU KNN
_, idx, nn = knn_points(pts, pts, K=k, return_nn=True)  # (B, N, K, 3)
centered = nn - pts.unsqueeze(2)
cov = torch.bmm(centered.transpose(-1,-2), centered) / k  # (B*N, 3, 3)
eigenvalues = torch.linalg.eigvalsh(cov)  # fully batched, GPU
# Derive features from eigenvalues...
```

PyTorch3D's CUDA KNN handles ~50K points per batch element efficiently; larger clouds require tiling. **FAISS GPU** is an alternative KNN backend that scales better to millions of points. Both approaches feed into `torch.linalg.eigvalsh`, which handles batched 3×3 eigendecomposition on GPU. The pipeline is differentiable, making it compatible with downstream neural networks.

**cuPCL** (`NVIDIA-AI-IOT/cuPCL`) ports only registration, filtering, segmentation, and clustering to CUDA—**no eigenfeature or covariance computation**. It ships as precompiled `.so` files with minimal maintenance (last update ~2019). **NVIDIA Kaolin** focuses on neural rendering and meshes with no KNN for unstructured point clouds. **RAPIDS cuML** provides GPU PCA and KNN, but PCA operates on a single global matrix, not per-point neighborhoods.

---

## Deep learning preprocessors do not compute classical eigenfeatures

**RandLA-Net** computes Local Spatial Encoding (LocSE) features: relative positions `(pᵢ - pᵢᵏ)` and Euclidean distances `‖pᵢ - pᵢᵏ‖` concatenated into a 10D vector per neighbor, then processed by an MLP. **No eigenvalues, covariance matrices, or classical geometric features** are computed. Preprocessing uses CPU-based scipy/nanoflann KDTree for KNN.

**KPConv** takes raw coordinates (optionally with colors/normals) as input. Its kernel-point convolutions learn geometric features implicitly from 3D positions—**no eigenfeature preprocessing exists**. The C++ extensions provide grid subsampling and radius search (via nanoflann) but no covariance computation. Neither pipeline's preprocessing is usable as a standalone eigenfeature extractor.

---

## Comprehensive comparison across all libraries

| Library | Single-pass multi-feature | Core-point support | Threading | Python bindings | GPU | Maintenance | Scale (10M+ pts) |
|---|---|---|---|---|---|---|---|
| **pgeof** | ✅ All in one call | ✅ Via CSR neighborhoods | Taskflow (parallel) | ✅ nanobind | ❌ CPU only | ✅ Active | ✅ Good |
| **PDAL covariancefeatures** | ✅ `feature_set: "all"` | ⚠️ `where` clause only | OpenMP (`threads` param) | ✅ Good | ❌ CPU only | ✅ Active (v2.10) | ✅ Good |
| **jakteristics** | ✅ All radius-based | ❌ All points | OpenMP | ✅ pip install | ❌ CPU only | ⚠️ Moderate | ⚠️ Millions OK |
| **PCL** | ❌ Separate passes per estimator | ✅ setSearchSurface | OpenMP / TBB | ❌ Unmaintained | ❌ CPU only | ✅ Active (v1.15) | ⚠️ In-memory |
| **Open3D** | ❌ Covariance → numpy eigendecomp | ❌ No search surface | OpenMP / TBB | ✅ Excellent | ⚠️ Normals only | ✅ Active (v0.19) | ⚠️ Memory-heavy |
| **CGAL Classification** | ✅ Local_eigen_analysis | ❌ | TBB | ⚠️ Fragile SWIG | ❌ CPU only | ✅ Active | ✅ Production |
| **PyTorch3D + eigh** | ✅ Custom pipeline | ✅ Separate query/search tensors | CUDA | ✅ Native | ✅ Full GPU | ✅ Active | ⚠️ Needs tiling |
| **FAISS GPU + PyTorch** | ✅ Custom pipeline | ✅ search vs. query | CUDA | ✅ Native | ✅ Full GPU | ✅ Active | ✅ Millions |
| **nanoflann + Eigen** | ✅ Custom C++ loop | ✅ Any query set | Thread-safe queries | ❌ No official | ❌ CPU only | ✅ Active (v1.10) | ✅ Fastest CPU |
| **libpointmatcher** | ⚠️ Eigenvalues as byproduct | ❌ ICP-focused | Single-threaded | ⚠️ pypointmatcher | ❌ | ✅ (Norlab) | ❌ Not designed for this |
| **py4dgeo** | ❌ M3C2 only | ✅ Core points (M3C2) | OpenMP | ✅ Primary | ❌ | ✅ Active | ❌ No eigenfeatures |
| **CloudCompare CLI** | ❌ One feature per pass | ❌ | Limited | ⚠️ CloudComPy | ❌ | ✅ Active | ❌ Slow for batch |
| **laspy + scipy** | ✅ Full custom | ✅ Any query set | `workers` param | ✅ Native | ❌ | ✅ (ecosystem) | ⚠️ Slow inner loop |

---

## Practical recommendations for terrain classification pipelines

**For production Python pipelines (recommended):** Use **pgeof** as the primary eigenfeature engine. It directly implements Weinmann's optimal neighborhood selection, computes all features in a single pass with C++ parallelism, and supports core-point subsampling via CSR neighborhoods. Combine with **laspy** for I/O and scikit-learn or LightGBM for classification. This delivers the flexibility and performance the user wants beyond CloudCompare.

**For PDAL-centric workflows:** Use `filters.covariancefeatures` with `feature_set: "all"` and `threads: N` as a single filter. Chain with `filters.optimalneighborhood` if Weinmann-style adaptive *k* is needed. This is the easiest path if you're already using PDAL pipelines for ground filtering (SMRF/CSF) and want to add eigenfeatures.

**For GPU acceleration at scale:** Build a PyTorch3D or FAISS-GPU pipeline with `torch.linalg.eigvalsh` for batched 3×3 eigendecomposition. This is the only path to processing tens of millions of points in seconds on a single GPU. Requires ~20 lines of custom code but no external C++ compilation.

**For maximum C++ performance:** Combine **nanoflann** (header-only KNN, thread-safe queries) with **Eigen** (3×3 eigendecomposition) in a custom OpenMP-parallelized loop. This is the fastest possible CPU path and gives full control over core-point subsampling, multi-scale computation, and memory layout. pgeof essentially implements this pattern already, so start there and customize if needed.

**Avoid** FLANN (abandoned), cuPCL (no eigenfeatures), and python-pcl/pclpy (unmaintained bindings). Deep learning preprocessors (RandLA-Net, KPConv) compute learned or relative features, not classical eigenfeatures. py4dgeo is M3C2-only. CloudCompare's per-feature independent computation confirms the user's concern—it is genuinely slower than single-pass alternatives for batch multi-feature extraction.