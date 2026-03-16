# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a vegetation filtering project for 3D LiDAR point clouds. It uses `cloudComPy` (Python bindings for CloudCompare) to compute per-point Gaussian curvature and 3D volume density features at multiple radii, then derives a curvature-density ratio to distinguish vegetation from other surfaces.

## Environment

The code runs in a Jupyter notebook (`test.ipynb`) using `cloudComPy` — a Python binding for CloudCompare. This library must be installed/available in the Python environment; it is not installable via pip and typically requires a pre-built CloudCompare+cloudComPy environment.

To run the notebook:
```bash
jupyter notebook test.ipynb
# or
jupyter lab
```

## Data

- Input point clouds are `.bin` files (CloudCompare binary format) stored under `data/` (gitignored)
- The primary test input is `data/gw.bin`; the production scans are at `/data/Scans/E/Alignments/`
- Output is also written as `.bin` files (e.g., `test_output.bin`)

## Algorithm

The core pipeline in `test.ipynb`:
1. Load a full point cloud and extract a local bounding-box subset around a test point
2. Spatially resample the subset at 1 cm resolution
3. Compute Gaussian curvature and 3D volume density at multiple paired radii (`curv_radii`, `dens3d_radii`)
4. Normalize each feature to [0,1] and compute `ratio = curvature - density` per radius pair
5. Sum ratios across all scales → `curv_density_ratio` scalar field
6. Apply bilateral filter on that scalar field
7. Save result via `ccp.SavePointCloud`

The intuition: vegetation tends to have high curvature and low density relative to solid surfaces, so a positive ratio indicates likely vegetation.
