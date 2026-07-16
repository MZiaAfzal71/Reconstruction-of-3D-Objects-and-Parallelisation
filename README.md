# Cross-Sectional 3D Surface Reconstruction as a Scientific Python Benchmark: Comparing NumPy, Numba, and PyTorch on CPUs and GPUs

This repository contains the source code, benchmark drivers, and generated results accompanying the manuscript:

> **Cross-Sectional 3D Surface Reconstruction as a Scientific Python Benchmark: Comparing NumPy, Numba, and PyTorch on CPUs and GPUs**

The repository provides a complete implementation of the proposed cross-sectional 3D surface reconstruction algorithm together with multiple Python implementations designed for benchmarking computational performance on CPUs and GPUs.

---

## Repository Structure

```
.
├── code/
├── driver scripts/
├── results/
└── README.md
```

### `code/`

Contains the complete source code implementing the reconstruction algorithm and all benchmark programs.

This directory includes:

* Pure Python Numpy-Assisted (nested loops) implementation
* NumPy vectorized implementation
* Numba CPU implementation
* Numba parallel CPU implementation
* Numba CUDA GPU implementation
* PyTorch CPU implementation
* PyTorch CUDA GPU implementation
* Precision analysis (float32 vs. float64)
* Continuity analysis
* Peak memory measurement utilities
* Plot generation utilities
* Supporting functions used by all implementations

---

### `driver scripts/`

Contains scripts that automate the execution of the benchmark suite.

The available drivers are:

* **generate_results_cpu.py**

  Executes every CPU benchmark, including

  * Pure Python
  * NumPy
  * Numba
  * Numba Parallel
  * PyTorch CPU
  * Precision and continuity analysis
  * Peak memory measurements

* **generate_results_gpu.py**

  Executes every GPU benchmark, including

  * Numba CUDA
  * PyTorch CUDA

* **aggregate_results_generate_plots.py**

  Aggregates all benchmark results and generates every table and figure used in the manuscript.

---

### `results/`

Contains the generated benchmark outputs.

Typical contents include

* CSV files containing benchmark statistics
* PDF figures
* Tables used in the manuscript
* Precision and continuity analysis results

The repository already contains the generated benchmark results, allowing users to reproduce the manuscript figures without rerunning the complete benchmark suite.

---

# Running the Repository

## 1. Clone the repository

```bash
git clone https://github.com/MZiaAfzal71/Reconstruction-of-3D-Objects-and-Parallelisation.git

cd Reconstruction-of-3D-Objects-and-Parallelisation
```

---

## 2. Generate CPU benchmark results

```bash
python "driver scripts/generate_results_cpu.py"
```

This script

* executes all CPU implementations,
* performs precision and continuity analysis,
* measures peak memory usage, and
* saves benchmark results under the `results/` directory.

---

## 3. Generate GPU benchmark results

```bash
python "driver scripts/generate_results_gpu.py"
```

**A CUDA-enabled GPU is required.**

This script benchmarks

* Numba CUDA
* PyTorch CUDA

and stores the generated benchmark statistics under `results/`.

---

## 4. Generate manuscript tables and figures

```bash
python "driver scripts/aggregate_results_generate_plots.py"
```

This script

* aggregates the CPU and GPU benchmark results,
* computes summary statistics, and
* generates all publication-quality tables and figures reported in the manuscript.

Since the repository already contains the benchmark results, this step can be executed directly to reproduce the manuscript figures without rerunning the benchmark experiments.

---

# Hardware Requirements

## CPU

Any modern multi-core CPU supported by Python.

## GPU (optional)

A CUDA-enabled NVIDIA GPU is required for executing the GPU benchmarks.

---

# Software Requirements

The project is implemented in Python and uses

* NumPy
* Pandas
* Matplotlib
* Numba
* PyTorch
* SciPy

---

# Output

Running the benchmark suite generates

* execution time statistics,
* memory usage statistics,
* precision analysis,
* continuity analysis,
* benchmark comparison tables,
* publication-quality figures,
* CSV files used throughout the manuscript.

---

