"""Inference CLI for the vegetation filter Random Forest pipeline.

Usage:
    python predict.py input.las models/rf_veg.joblib \\
        --output input_predicted.las \\
        --core-voxel-size 0.1 \\
        --write-probabilities
"""
import argparse

import joblib
import numpy as np
from scipy.spatial import KDTree

from feature_extraction import extract_features_tiled, spatial_subsample
from io_utils import get_scalar_field, load_las, write_las_with_prediction, rgb_to_lab_ab, _LAB_FIELDS


def propagate_to_full_cloud(core_preds, xyz_core, xyz_full) -> np.ndarray:
    """Assign each full-cloud point the label of its nearest core point."""
    tree = KDTree(xyz_core)
    _, idx = tree.query(xyz_full, k=1, workers=-1)
    return core_preds[idx]


def predict_las(
    las_path,
    model_path,
    output_path,
    core_voxel_size=None,
    tile_size=20.0,
    pred_field_name="PredictedClass",
    write_probabilities=False,
):
    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    feature_names = bundle["feature_names"]
    radii = bundle["radii"]
    scalar_field_names = bundle["scalar_field_names"]

    print(f"Loading {las_path} ...", flush=True)
    xyz, las = load_las(las_path)

    lab_cache = {}
    if any(f in _LAB_FIELDS for f in scalar_field_names):
        try:
            lab_cache["lab_a"], lab_cache["lab_b"] = rgb_to_lab_ab(las)
        except Exception as e:
            print(f"  Warning: LAB conversion failed ({e}) — using zeros")
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
                print(f"  Warning: {e} — using zeros for '{fname}'")
                scalar_fields[fname] = np.zeros(len(xyz), dtype=np.float32)

    if core_voxel_size is not None:
        core_idx = spatial_subsample(xyz, core_voxel_size)
        xyz_core = xyz[core_idx]
        print(f"Core points: {len(xyz_core)} / {len(xyz)} total")
    else:
        core_idx = None
        xyz_core = xyz

    print(f"Extracting features for {len(xyz_core)} points ...", flush=True)
    X = extract_features_tiled(
        xyz=xyz_core,
        xyz_search=xyz,
        radii=radii,
        scalar_fields=scalar_fields,
        tile_size=tile_size,
    )

    if list(X.columns) != feature_names:
        raise ValueError(
            f"Feature column mismatch!\n"
            f"Expected: {feature_names}\n"
            f"Got:      {list(X.columns)}"
        )

    core_preds = pipeline.predict(X)
    probs = None
    if write_probabilities:
        probs_all = pipeline.predict_proba(X)
        classes = pipeline.classes_
        # Use max probability across all classes as confidence, or veg-class prob
        # if binary (class 1 assumed vegetation)
        if len(classes) == 2:
            veg_col = np.where(classes == 1)[0]
            if len(veg_col):
                probs = probs_all[:, veg_col[0]]
            else:
                probs = probs_all.max(axis=1)
        else:
            probs = probs_all.max(axis=1)

    if core_idx is not None:
        print("Propagating labels to full cloud ...", flush=True)
        predictions = propagate_to_full_cloud(core_preds, xyz_core, xyz)
        if probs is not None:
            probs = propagate_to_full_cloud(probs, xyz_core, xyz)
    else:
        predictions = core_preds

    write_las_with_prediction(
        las, predictions, output_path,
        field_name=pred_field_name,
        probabilities=probs,
        prob_field_name="VegProbability",
    )
    print(f"Written to {output_path}")

    classes, counts = np.unique(predictions, return_counts=True)
    for cls, cnt in zip(classes, counts):
        print(f"  Class {cls}: {cnt} ({100 * cnt / len(predictions):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Predict vegetation labels")
    parser.add_argument("las_file")
    parser.add_argument("model_path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--core-voxel-size", type=float, default=None)
    parser.add_argument("--pred-field-name", default="PredictedClass")
    parser.add_argument("--write-probabilities", action="store_true")
    args = parser.parse_args()

    predict_las(
        las_path=args.las_file,
        model_path=args.model_path,
        output_path=args.output,
        core_voxel_size=args.core_voxel_size,
        pred_field_name=args.pred_field_name,
        write_probabilities=args.write_probabilities,
    )


if __name__ == "__main__":
    main()
