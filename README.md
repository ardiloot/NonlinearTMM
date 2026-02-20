[![PyPI version](https://badge.fury.io/py/NonlinearTMM.svg)](https://badge.fury.io/py/NonlinearTMM)
[![Pytest](https://github.com/ardiloot/NonlinearTMM/actions/workflows/pytest.yml/badge.svg)](https://github.com/ardiloot/NonlinearTMM/actions/workflows/pytest.yml)
[![PyPI](https://github.com/ardiloot/NonlinearTMM/actions/workflows/publish.yml/badge.svg)](https://github.com/ardiloot/NonlinearTMM/actions/workflows/publish.yml)
[![Pre-commit](https://github.com/ardiloot/NonlinearTMM/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/ardiloot/NonlinearTMM/actions/workflows/pre-commit.yml)

# NonlinearTMM: Nonlinear transfer-matrix method

## Overview

Transfer-matrix method (TMM) is a powerful analytical method to solve Maxwell's
equations in layered structures. However, standard TMM is limited to infinite plane
waves (e.g. no Gaussian beam excitation) and to linear processes (i.e. calculation
of second-harmonic, sum-frequency, and difference-frequency generation is not
possible). The aim of this package is to extend standard TMM to include those
features. The physics of these extensions is described in the following
publications:

1. [A. Loot and V. Hizhnyakov, "Extension of standard transfer-matrix method for three-wave mixing for plasmonic structures," Appl. Phys. A, vol. 123, no. 3, p. 152, 2017.](https://link.springer.com/article/10.1007%2Fs00339-016-0733-0)
2. [A. Loot and V. Hizhnyakov, "Modeling of enhanced spontaneous parametric down-conversion in plasmonic and dielectric structures with realistic waves," J. Opt., vol. 20, no. 055502, 2018.](https://doi.org/10.1088/2040-8986/aab6c0)

For additional details, see the [documentation](https://ardiloot.github.io/NonlinearTMM/).
For a getting started guide, see [Getting started](https://ardiloot.github.io/NonlinearTMM/GettingStarted.html).

## Main features

In addition to the standard TMM features, this package also supports:

* Calculation of Gaussian beam (or any other beam) propagation inside layered structures
* Calculation of nonlinear processes: SHG, SFG, DFG

## Technical features

* Core written in C++
* Python bindings via Cython
* OpenMP parallelization (Linux and Windows)
* Supports Linux (x86_64), Windows (x64, ARM64), and macOS (ARM64)

## Installation

Requires Python >= 3.10.

```bash
pip install NonlinearTMM
```

## Quick example

```python
import math
import numpy as np
from NonlinearTMM import TMM, Material

# Define materials
prism = Material.Static(1.5)
ag = Material.Static(0.054007 + 3.4290j)  # Silver @ 532nm
air = Material.Static(1.0)

# Set up TMM (Kretschmann configuration)
tmm = TMM(wl=532e-9, pol="p", I0=1.0)
tmm.AddLayer(math.inf, prism)
tmm.AddLayer(50e-9, ag)
tmm.AddLayer(math.inf, air)

# Sweep angle of incidence
betas = np.sin(np.radians(np.linspace(0, 80, 500))) * 1.5
result = tmm.Sweep("beta", betas)

print(f"Min reflectance: {result.Ir.min():.4f}")
```

## Documentation

https://ardiloot.github.io/NonlinearTMM/

## Development

```bash
# Clone with submodules (Eigen)
git clone --recurse-submodules https://github.com/ardiloot/NonlinearTMM.git
cd NonlinearTMM

# Install dev environment
uv sync

# Run tests
uv run pytest

# Run pre-commit checks
uv run pre-commit run --all-files
```

## License

MIT
