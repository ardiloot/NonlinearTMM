# NonlinearTMM : Nonlinear transfer-matrix method

## Overview

Transfer-matrix method (TMM) is powerful analytical method to solve Maxwell equations in layered structures. However, standard TMM is limited by infinite plane waves (e.g no Gaussian beam excitation) and it is only limited to linear processes (i.e calculation of second-harmonic, sum-frequency, difference-frequency generation is not possible). The aim of this package is extand standard TMM to include those features. The physics of those extensions are described in the follwoing publications, first extends the standard TMM to nonlinear processes and the second extends to the beams with arbritary profiles.

1. [A. Loot and V. Hizhnyakov, “Extension of standard transfer-matrix method for three-wave mixing for plasmonic structures,” Appl. Phys. A, vol. 123, no. 3, p. 152, 2017.](https://link.springer.com/article/10.1007%2Fs00339-016-0733-0)
2. TBP

For additional details see our documentation https://ardiloot.github.io/NonlinearTMM/. For getting started guide see [Getting started](https://ardiloot.github.io/NonlinearTMM/GettingStarted.html).

## Main features

In addition to the standard TMM features this package also supports:

* Calculation of Gaussian beam (or any other beam) propagartion inside layered structures
* Calculation of nonlinear processes SHG/SFG/DFG

## Technical features

* Written in C++
* Python wrapper written in Cython
* Parallerization through OpenMP
* Use of SSE instructions for speedup

## Documentation

https://ardiloot.github.io/NonlinearTMM/
