"""Training CLI for the vegetation filter Random Forest pipeline.

Usage:
    python train.py site_a.las site_b.las \\
        --radii 0.5 1.0 2.0 \\
        --scalar-fields intensity return_number number_of_returns red green blue \\
        --label-field Label \\
        --veg-class 0 \\
        --unlabeled-value -1 \\
        --core-voxel-size 0.1 \\
        --tile-size 20 \\
        --output models/rf_veg.joblib
"""
import argparse
import hashlib
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from feature_extraction import extract_features_tiled, spatial_subsample
from io_utils import get_labels, get_scalar_field, load_las, rgb_to_lab_ab, _LAB_FIELDS


def _feature_cache_path(cache_dir, path, radii, scalar_field_names, k_min,
                         training_densities, max_points_per_density, tile_size):
    """Return the .npz cache path for a given file + parameter combination."""
    stat = os.stat(path)
    key = json.dumps({
        "path": os.path.abspath(path),
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "radii": sorted(radii),
        "scalar_field_names": sorted(scalar_field_names),
        "k_min": k_min,
        "training_densities": sorted(training_densities) if training_densities else None,
        "max_points_per_density": max_points_per_density,
        "tile_size": tile_size,
    }, sort_keys=True)
    digest = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(cache_dir, f"{digest}.npz")


def _load_file_features(
    path, radii, scalar_field_names, label_field,
    unlabeled_value, veg_class, training_densities, k_min, tile_size,
    max_points_per_density=None, rng=None, cache_dir=None, n_jobs=4,
):
    """Load one LAS file and return (X DataFrame, y_binary int32 array).

    training_densities: list of voxel sizes to subsample at, or [None] for
    no subsampling. Features from all densities are stacked row-wise so the
    model sees the same geometry at multiple point spacings.
    """
    # --- cache check ---
    if cache_dir is not None:
        cache_path = _feature_cache_path(
            cache_dir, path, radii, scalar_field_names, k_min,
            training_densities, max_points_per_density, tile_size,
        )
        if os.path.exists(cache_path):
            print(f"  [cache] {os.path.basename(path)} — loading from cache", flush=True)
            data = np.load(cache_path, allow_pickle=False)
            col_names = json.loads(data["col_names_json"].item())
            X = pd.DataFrame(data["X"], columns=col_names)
            y = data["y"]
            return X, y

    print(f"  Loading {path} ...", flush=True)
    xyz, las = load_las(path)
    labels_raw = get_labels(las, label_field)

    # Pre-compute LAB a/b once if any lab field is requested
    lab_cache = {}
    if any(f in _LAB_FIELDS for f in scalar_field_names):
        try:
            lab_cache["lab_a"], lab_cache["lab_b"] = rgb_to_lab_ab(las)
        except Exception as e:
            print(f"    Warning: LAB conversion failed ({e}) — using zeros")
            lab_cache["lab_a"] = np.zeros(len(xyz), dtype=np.float32)
            lab_cache["lab_b"] = np.zeros(len(xyz), dtype=np.float32)

    scalar_fields = {}
    for fname in scalar_field_names:
        if fname in _LAB_FIELDS:
            scalar_fields[fname] = lab_cache[fname]
        else:
            try:
                scalar_fields[fname] = get_scalar_field(las, fname)
            except KeyError as e:
                print(f"    Warning: {e} — using zeros for '{fname}'")
                scalar_fields[fname] = np.zeros(len(xyz), dtype=np.float32)

    labeled_mask = labels_raw != unlabeled_value
    n_labeled = labeled_mask.sum()
    print(
        f"    {len(xyz):,} total pts, {n_labeled:,} labeled "
        f"({100 * n_labeled / max(len(xyz), 1):.1f}%)"
    )
    if n_labeled == 0:
        return None, None

    xyz_labeled = xyz[labeled_mask]
    labels_labeled = labels_raw[labeled_mask]

    X_densities, y_densities = [], []
    for voxel_size in training_densities:
        if voxel_size is not None:
            core_idx = spatial_subsample(xyz_labeled, voxel_size)
            xyz_core = xyz_labeled[core_idx]
            y_core = labels_labeled[core_idx]
        else:
            xyz_core = xyz_labeled
            y_core = labels_labeled

        if max_points_per_density is not None and len(xyz_core) > max_points_per_density:
            if rng is None:
                rng = np.random.default_rng(42)
            sel = rng.choice(len(xyz_core), max_points_per_density, replace=False)
            xyz_core = xyz_core[sel]
            y_core = y_core[sel]

        y_binary = (y_core == veg_class).astype(np.int32)
        tag = f"{voxel_size}m" if voxel_size else "all"
        print(
            f"    [{tag}] {len(xyz_core):,} core pts  |  "
            f"veg={y_binary.sum():,}  non-veg={(y_binary==0).sum():,}",
            flush=True,
        )

        X_d = extract_features_tiled(
            xyz=xyz_core,
            xyz_search=xyz,
            radii=radii,
            scalar_fields=scalar_fields,
            k_min=k_min,
            tile_size=tile_size,
            n_jobs=n_jobs,
        )
        X_densities.append(X_d)
        y_densities.append(y_binary)

    X = pd.concat(X_densities, ignore_index=True)
    y = np.concatenate(y_densities)

    # --- cache save ---
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(
            cache_path,
            X=X.values.astype(np.float32),
            y=y,
            col_names_json=np.array(json.dumps(list(X.columns))),
        )
        print(f"    [cache] saved to {cache_path}", flush=True)

    return X, y


def collect_training_data(
    las_paths, radii, scalar_field_names,
    label_field="Label", unlabeled_value=-1, veg_class=0,
    training_densities=None, k_min=3, tile_size=20.0,
    max_points_per_density=None, cache_dir=None, n_jobs=4,
):
    """Load all files, extract features, return (X, y, file_ids).

    training_densities: list of voxel sizes (e.g. [0.05, 0.1, 0.15, 0.2]).
    Each file is subsampled at every density; rows are stacked.
    Pass [None] or None for no subsampling.
    file_ids is an int array of file index per row — used for LOFO CV.
    """
    if training_densities is None:
        training_densities = [None]

    X_parts, y_parts, fid_parts = [], [], []
    feature_names = None

    for fid, path in enumerate(las_paths):
        X_file, y_file = _load_file_features(
            path, radii, scalar_field_names, label_field,
            unlabeled_value, veg_class, training_densities, k_min, tile_size,
            max_points_per_density=max_points_per_density,
            cache_dir=cache_dir,
            n_jobs=n_jobs,
        )
        if X_file is None:
            continue

        if feature_names is None:
            feature_names = list(X_file.columns)
        elif list(X_file.columns) != feature_names:
            raise ValueError(
                f"Feature columns for {path} do not match previous files.\n"
                f"Expected: {feature_names}\nGot: {list(X_file.columns)}"
            )

        X_parts.append(X_file.values.astype(np.float32))
        y_parts.append(y_file)
        fid_parts.append(np.full(len(y_file), fid, dtype=np.int32))

    if not X_parts:
        raise RuntimeError("No training data collected.")

    X = pd.DataFrame(np.vstack(X_parts), columns=feature_names)
    y = np.concatenate(y_parts)
    file_ids = np.concatenate(fid_parts)

    print(f"\nTotal: {len(y):,} pts  |  veg={y.sum():,}  non-veg={(y==0).sum():,}")
    return X, y, file_ids


def train_model(X, y, n_estimators=150, max_depth=25,
                class_weight="balanced", n_jobs=-1, random_state=42) -> Pipeline:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            oob_score=True,
            n_jobs=n_jobs,
            random_state=random_state,
        )),
    ])
    pipe.fit(X, y)
    print(f"OOB accuracy: {pipe.named_steps['rf'].oob_score_:.4f}")
    return pipe


def save_model(pipeline, feature_names, radii, scalar_field_names, veg_class, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "feature_names": feature_names,
        "radii": radii,
        "scalar_field_names": scalar_field_names,
        "veg_class": veg_class,
    }, output_path)
    print(f"Model saved to {output_path}")


def lofo_cross_validation(las_paths, radii, scalar_field_names,
                           label_field, unlabeled_value, veg_class,
                           training_densities, k_min, tile_size,
                           max_points_per_density=None,
                           n_estimators=150, max_depth=25, cache_dir=None, n_jobs=4):
    """Leave-one-file-out cross-validation. Prints per-fold and overall metrics."""
    print("\n=== Leave-One-File-Out Cross-Validation ===")

    # Pre-extract features for all files
    all_X, all_y, file_ids = collect_training_data(
        las_paths, radii, scalar_field_names,
        label_field=label_field, unlabeled_value=unlabeled_value,
        veg_class=veg_class, training_densities=training_densities,
        k_min=k_min, tile_size=tile_size,
        max_points_per_density=max_points_per_density,
        cache_dir=cache_dir,
        n_jobs=n_jobs,
    )

    unique_fids = np.unique(file_ids)
    all_preds = np.zeros(len(all_y), dtype=np.int32)
    fold_results = []

    for fold, held_out in enumerate(unique_fids):
        fname = os.path.basename(las_paths[held_out])
        train_mask = file_ids != held_out
        test_mask = file_ids == held_out

        X_tr, y_tr = all_X[train_mask], all_y[train_mask]
        X_te, y_te = all_X[test_mask], all_y[test_mask]

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("rf", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )),
        ])
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        all_preds[test_mask] = preds

        acc = (preds == y_te).mean()
        n_veg_test = y_te.sum()
        fold_results.append((fname, acc, len(y_te), n_veg_test))
        print(f"  Fold {fold+1:2d} ({fname}): acc={acc:.4f}  "
              f"test_pts={len(y_te):,}  veg={n_veg_test:,}")

    print("\nOverall (all folds):")
    print(classification_report(all_y, all_preds, target_names=["non-veg", "veg"]))
    cm = confusion_matrix(all_y, all_preds)
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=["true non-veg", "true veg"],
                       columns=["pred non-veg", "pred veg"]).to_string())
    return fold_results, all_preds


def main():
    parser = argparse.ArgumentParser(description="Train vegetation RF classifier")
    parser.add_argument("las_files", nargs="+")
    parser.add_argument("--radii", nargs="+", type=float, required=True)
    parser.add_argument("--scalar-fields", nargs="*", default=[], dest="scalar_fields")
    parser.add_argument("--label-field", default="Label")
    parser.add_argument("--unlabeled-value", type=int, default=-1)
    parser.add_argument("--veg-class", type=int, default=0,
                        help="Label value for vegetation (default: 0)")
    parser.add_argument("--training-densities", nargs="+", type=float, default=None,
                        dest="training_densities",
                        help="Voxel sizes to subsample training data at (e.g. 0.1 0.15 0.2 0.3). "
                             "Each file is subsampled at every density and rows are stacked. "
                             "Omit for no subsampling.")
    parser.add_argument("--max-points-per-density", type=int, default=None,
                        dest="max_points_per_density",
                        help="Cap core points per file per density level (random subsample). "
                             "Keeps total training set manageable (e.g. 8000 → ~450K max total).")
    parser.add_argument("--tile-size", type=float, default=20.0,
                        help="XY tile size in metres for tiled KDTree (default: 20)")
    parser.add_argument("--n-jobs", type=int, default=4, dest="n_jobs",
                        help="Parallel workers for tile processing (default: 8). "
                             "Each worker uses ~1-2 GB RAM per worker; increase if you have spare RAM (rule of thumb: free_GB / 2).")
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=25)
    parser.add_argument("--output", default="models/rf_veg.joblib")
    parser.add_argument("--lofo", action="store_true",
                        help="Run leave-one-file-out CV before final training")
    parser.add_argument("--cache-dir", default="cache/", dest="cache_dir",
                        help="Directory to cache per-file features (default: cache/). "
                             "Pass empty string to disable.")
    args = parser.parse_args()

    training_densities = args.training_densities  # None means [None] inside collect_training_data
    cache_dir = args.cache_dir if args.cache_dir else None

    if args.lofo:
        lofo_cross_validation(
            las_paths=args.las_files,
            radii=args.radii,
            scalar_field_names=args.scalar_fields,
            label_field=args.label_field,
            unlabeled_value=args.unlabeled_value,
            veg_class=args.veg_class,
            training_densities=training_densities,
            k_min=args.k_min,
            tile_size=args.tile_size,
            max_points_per_density=args.max_points_per_density,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            cache_dir=cache_dir,
            n_jobs=args.n_jobs,
        )
        print()

    print("=== Training final model on all data ===")
    X, y, _ = collect_training_data(
        las_paths=args.las_files,
        radii=args.radii,
        scalar_field_names=args.scalar_fields,
        label_field=args.label_field,
        unlabeled_value=args.unlabeled_value,
        veg_class=args.veg_class,
        training_densities=training_densities,
        k_min=args.k_min,
        tile_size=args.tile_size,
        max_points_per_density=args.max_points_per_density,
        cache_dir=cache_dir,
        n_jobs=args.n_jobs,
    )
    pipeline = train_model(X, y,
                           n_estimators=args.n_estimators,
                           max_depth=args.max_depth)
    save_model(pipeline, list(X.columns), args.radii, args.scalar_fields,
               args.veg_class, args.output)


if __name__ == "__main__":
    main()
