.. py:currentmodule:: NonlinearTMM

Introduction
############

Overview
********
Transfer-matrix method (TMM) is a powerful analytical method to solve Maxwell's
equations in layered structures. However, standard TMM is limited to infinite
plane waves (e.g. no Gaussian beam excitation) and to linear processes (i.e.
calculation of second-harmonic, sum-frequency, and difference-frequency
generation is not possible). The aim of this package is to extend standard TMM to
include those features. The physics of these extensions are described in the
following publications:

1. `A. Loot and V. Hizhnyakov, "Extension of standard transfer-matrix method for three-wave mixing for plasmonic structures," Appl. Phys. A, vol. 123, no. 3, p. 152, 2017. <https://link.springer.com/article/10.1007%2Fs00339-016-0733-0>`_
2. `A. Loot and V. Hizhnyakov, "Modeling of enhanced spontaneous parametric down-conversion in plasmonic and dielectric structures with realistic waves," J. Opt., vol. 20, no. 055502, 2018. <https://doi.org/10.1088/2040-8986/aab6c0>`_


Main features
=============

In addition to the standard TMM features this package also supports:

* Calculation of Gaussian beam (or any other beam) propagation inside layered structures
* Calculation of nonlinear processes SHG/SFG/DFG

Technical features
==================

* Core written in C++
* Python bindings via Cython
* OpenMP parallelization (Linux and Windows)
* Supports Linux (x86_64), Windows (x64, ARM64), and macOS (ARM64)
