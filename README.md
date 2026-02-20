[![PyPI version](https://badge.fury.io/py/NonlinearTMM.svg)](https://badge.fury.io/py/NonlinearTMM)
[![Python](https://img.shields.io/pypi/pyversions/NonlinearTMM)](https://pypi.org/project/NonlinearTMM/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pytest](https://github.com/ardiloot/NonlinearTMM/actions/workflows/pytest.yml/badge.svg)](https://github.com/ardiloot/NonlinearTMM/actions/workflows/pytest.yml)
[![Pre-commit](https://github.com/ardiloot/NonlinearTMM/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/ardiloot/NonlinearTMM/actions/workflows/pre-commit.yml)
[![Build and upload to PyPI](https://github.com/ardiloot/NonlinearTMM/actions/workflows/publish.yml/badge.svg)](https://github.com/ardiloot/NonlinearTMM/actions/workflows/publish.yml)

# NonlinearTMM: Nonlinear Transfer-Matrix Method

A Python library for optical simulations of **multilayer structures** using the transfer-matrix method, extended to support **nonlinear processes** (SHG, SFG, DFG) and **Gaussian beam propagation**.

<p align="center">
  <img src="docs/images/TMMForWaves-example.png" alt="Gaussian beam exciting surface plasmon polaritons" width="700">
</p>

> **See also:** [GeneralTmm](https://github.com/ardiloot/GeneralTmm) — a 4×4 TMM for **anisotropic** (birefringent) multilayer structures.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [API Overview](#api-overview)
- [Examples](#examples)
  - [Surface Plasmon Polaritons](#surface-plasmon-polaritons--exampletmmpy)
  - [Gaussian Beam Excitation](#gaussian-beam-excitation--exampletmmforwavespy)
  - [Second-Harmonic Generation](#second-harmonic-generation--examplesecondordernonlineartmmpy)
- [References](#references)
- [Documentation](#documentation)
- [Development](#development)
  - [Setup](#setup)
  - [Running tests](#running-tests)
  - [Code formatting and linting](#code-formatting-and-linting)
  - [CI overview](#ci-overview)
- [Releasing](#releasing)
- [License](#license)

## Features

- **Standard TMM** — reflection, transmission, absorption for p- and s-polarized plane waves at arbitrary angles
- **Parameter sweeps** — over wavelength, angle of incidence, layer thickness, or any other parameter
- **1D and 2D electromagnetic field profiles** — E and H field distributions through the structure
- **Field enhancement** — calculation of field enhancement factors (e.g. for SPP excitation)
- **Gaussian beam propagation** — any beam profile through layered structures, not just plane waves
- **Second-order nonlinear processes** — SHG, SFG, DFG in multilayer structures
- **Wavelength-dependent materials** — interpolated from measured optical data (YAML format)
- **High performance** — C++ core (Eigen) with Cython bindings, OpenMP parallelization
- **Cross-platform wheels** — Linux (x86_64), Windows (x64, ARM64), macOS (ARM64); Python 3.10–3.13

## Installation

```bash
pip install NonlinearTMM
```

Pre-built wheels are available for most platforms. A C++ compiler is only needed when installing from source.

## API Overview

The library exposes three main classes: `Material`, `TMM`, and `SecondOrderNLTMM`.

| Class / method | Purpose |
|---|---|
| `Material(wls, ns)` | Wavelength-dependent material from arrays of λ and complex n |
| `Material.Static(n)` | Constant refractive index (shortcut) |
| `Material.FromFile(path)` | Load material from a YAML data file |
| `TMM(wl=…, pol=…, I0=…)` | Create a solver; `wl` = wavelength (m), `pol` = `"p"` or `"s"` |
| `tmm.AddLayer(d, mat)` | Append layer (`d` in m, `inf` for semi-infinite) |
| `tmm.Sweep(param, values)` | Solve for an array of values of any parameter |
| `tmm.GetFields(zs)` | E, H field profiles along the layer normal |
| `tmm.GetFields2D(zs, xs)` | E, H on a 2-D grid |
| `tmm.GetEnhancement(layerNr)` | Field enhancement in a given layer |
| `tmm.wave` | Access `_Wave` parameters for Gaussian beam calculations |
| `tmm.WaveSweep(param, values)` | Parameter sweep for beam calculations |
| `tmm.WaveGetFields2D(zs, xs)` | 2-D field map for beam excitation |
| `SecondOrderNLTMM(…)` | Second-order nonlinear TMM (SHG, SFG, DFG) |

For the full API, see the [reference documentation](https://ardiloot.github.io/NonlinearTMM/Reference.html).

## Examples

### Surface Plasmon Polaritons — [ExampleTMM.py](Examples/ExampleTMM.py)

Kretschmann configuration (prism | 50 nm Ag | air) at 532 nm. Demonstrates
reflection sweeps, field enhancement, and 1D/2D field visualization of surface
plasmon polaritons.

```python
import math
import numpy as np
from NonlinearTMM import TMM, Material

# Materials
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
result = tmm.Sweep("beta", betas, outEnh=True, layerNr=2)
```

<p align="center">
  <img src="docs/images/TMM-example.png" alt="SPP reflection, enhancement, and field profiles" width="700">
</p>

### Gaussian Beam Excitation — [ExampleTMMForWaves.py](Examples/ExampleTMMForWaves.py)

Same Kretschmann structure excited by a 10 mW Gaussian beam (waist 10 μm).
Shows how finite beam width affects resonance depth and field enhancement.

<p align="center">
  <img src="docs/images/TMMForWaves-example.png" alt="Gaussian beam SPP excitation" width="700">
</p>

### Second-Harmonic Generation — [ExampleSecondOrderNonlinearTmm.py](Examples/ExampleSecondOrderNonlinearTmm.py)

Second-harmonic generation in a nonlinear crystal, calculated with the
`SecondOrderNLTMM` class. Supports SHG, SFG, and DFG processes.

## References

> Loot, A., & Hizhnyakov, V. (2017). Extension of standard transfer-matrix method for three-wave mixing for plasmonic structures. *Applied Physics A*, 123(3), 152. [doi:10.1007/s00339-016-0733-0](https://link.springer.com/article/10.1007%2Fs00339-016-0733-0)
>
> Loot, A., & Hizhnyakov, V. (2018). Modeling of enhanced spontaneous parametric down-conversion in plasmonic and dielectric structures with realistic waves. *Journal of Optics*, 20, 055502. [doi:10.1088/2040-8986/aab6c0](https://doi.org/10.1088/2040-8986/aab6c0)

## Documentation

Full documentation is available at https://ardiloot.github.io/NonlinearTMM/.

- [Getting started](https://ardiloot.github.io/NonlinearTMM/GettingStarted.html) — installation, package structure, examples
- [API reference](https://ardiloot.github.io/NonlinearTMM/Reference.html) — complete class and method reference

## Development

### Setup

```bash
git clone --recurse-submodules https://github.com/ardiloot/NonlinearTMM.git
cd NonlinearTMM

# Install uv if not already installed:
# https://docs.astral.sh/uv/getting-started/installation/

# Create venv, build the C++ extension, and install all dependencies
uv sync
```

### Running tests

```bash
uv run pytest -v
```

### Code formatting and linting

[Pre-commit](https://pre-commit.com/) hooks are configured to enforce formatting (ruff, clang-format) and catch common issues. To install the git hook locally:

```bash
uvx pre-commit install
```

To run all checks manually:

```bash
uvx pre-commit run --all-files
```

### CI overview

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| [Pytest](.github/workflows/pytest.yml) | Push to `master` / PRs | Tests on {ubuntu, windows, macos} × Python 3.10 |
| [Pre-commit](.github/workflows/pre-commit.yml) | Push to `master` / PRs | Runs ruff, clang-format, ty, and other checks |
| [Publish to PyPI](.github/workflows/publish.yml) | Release published | Builds wheels + sdist via cibuildwheel, uploads to PyPI |
| [Publish docs](.github/workflows/publish_docs.yml) | Release published | Builds Sphinx docs and deploys to GitHub Pages |

## Releasing

Versioning is handled automatically by [setuptools-scm](https://github.com/pypa/setuptools-scm) from git tags.

1. **Ensure CI is green** on the `master` branch.
2. **Create a new release** on GitHub:
   - Go to [Releases](https://github.com/ardiloot/NonlinearTMM/releases) → **Draft a new release**
   - Create a new tag following [PEP 440](https://peps.python.org/pep-0440/) (e.g. `v1.2.0`)
   - Target the `master` branch (or a specific commit on master)
   - Click **Generate release notes** for auto-generated changelog
   - For pre-releases (e.g. `v1.2.0rc1`), check **Set as a pre-release** — these upload to TestPyPI instead of PyPI
3. **Publish the release** — the workflow builds wheels for Linux (x86_64), Windows (x64, ARM64), and macOS (ARM64) and uploads to [PyPI](https://pypi.org/project/NonlinearTMM/).

## License

[MIT](LICENSE)
