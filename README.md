# K-Means Clustering: Single-Core, Multi-Core (OpenMP), and Scikit-learn Benchmark

## Overview

This project provides a comprehensive performance analysis and benchmarking suite for K-means clustering, featuring:

- **Single-core implementation** in pure Python/NumPy
- **Multi-core implementation** in C with OpenMP, callable from Python via `ctypes`
- **Scikit-learn implementation** for industry-standard comparison

The suite includes benchmarking tools, visualization utilities, and an interactive dashboard for exploring clustering results and performance metrics.

## Project Structure

- `kmeans.py` — Main Python module with all K-means implementations, benchmarking, and visualization tools.
- `kmeans_openmp.c` — C source code for the multi-core (OpenMP) K-means implementation.
- `kmeans_openmp.so` — Compiled shared library for the multi-core implementation (macOS/Linux; use `.dll` for Windows).
- `kmeans.ipynb` — Jupyter notebook for running demos, benchmarks, and visualizations.
- `finalbackup/kmeans_notebook.ipynb` — Additional notebook with similar demos and benchmarking.
- `kmeans_benchmark_results.csv` — Example output of benchmark results.
- `FinalReport.tex` — Detailed technical report and analysis.
- `ppt.html` — Project presentation slides.
- `requirements.txt` — Python dependencies.

## Features

- **Multiple Implementations:** Compare single-core, multi-core (OpenMP), and Scikit-learn K-means.
- **Benchmarking:** Measure execution time, memory usage, energy estimation, and clustering quality (Silhouette, Calinski-Harabasz, Davies-Bouldin scores).
- **Visualization:** 2D/3D cluster plots, performance dashboards, and interactive widgets (Jupyter).
- **Resource Monitoring:** Track CPU and memory usage during clustering.
- **Scalability Analysis:** Evaluate performance across different dataset sizes and thread counts.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Compile the OpenMP Library

On macOS/Linux:
```bash
gcc -fopenmp -O3 -shared -o kmeans_openmp.so -fPIC kmeans_openmp.c
```

On Windows:
```bash
gcc -fopenmp -O3 -shared -o kmeans_openmp.dll -fPIC kmeans_openmp.c
```

### 3. Run the Jupyter Notebook

```bash
jupyter notebook kmeans.ipynb
```

Or use the notebook in `finalbackup/`.

### 4. Example Usage

In Python:
```python
from kmeans import KMeansSingle, KMeansMulti, KMeansSK, DataGenerator

# Generate synthetic data
X, y = DataGenerator.make_blobs_data(n_samples=10000, n_features=2, n_clusters=5, random_state=42)

# Single-core
model_single = KMeansSingle(n_clusters=5, random_state=42)
model_single.fit(X)

# Multi-core (OpenMP)
model_multi = KMeansMulti(n_clusters=5, random_state=42)
model_multi.fit(X)

# Scikit-learn
model_sklearn = KMeansSK(n_clusters=5, random_state=42)
model_sklearn.fit(X)
```

## Benchmark Results

See `kmeans_benchmark_results.csv` for detailed results. Example metrics:

| Method   | n_samples | Time (s) | Silhouette | Inertia   |
|----------|-----------|----------|------------|-----------|
| single   | 1000      | 0.0052   | 0.678      | 1873.25   |
| multi    | 1000      | 0.0061   | 0.678      | 1873.25   |
| sklearn  | 1000      | 0.0672   | 0.678      | 1873.43   |
| ...      | ...       | ...      | ...        | ...       |

## Interactive Dashboard

To launch the interactive dashboard in a Jupyter notebook:
```python
from kmeans import InteractiveDashboard
InteractiveDashboard().display()
```

## Dependencies

See `requirements.txt` for all dependencies. Key packages:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- plotly
- psutil
- ipywidgets
- notebook

## References

- See `FinalReport.tex` for a detailed technical report and analysis.
- See `ppt.html` for a project presentation.
