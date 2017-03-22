.. py:currentmodule:: NonlinearTMM

Introduction
############

Overview
********
Transfer-matrix method (TMM) is powerful analytical method to solve Maxwell 
equations in layered structures. However, standard TMM is limited by infinite 
plane waves (e.g no Gaussian beam excitation) and it is only limited to linear 
processes (i.e calculation of second-harmonic, sum-frequency, difference-frequency 
generation is not possible). The aim of this package is extand standard TMM to 
include those features. The physics of those extensionsare described in the 
follwoing publications, first extends the standard TMM to nonlinear processes 
and the second extends to the beams with arbritary profiles.

1. `A. Loot and V. Hizhnyakov, “Extension of standard transfer-matrix method for three-wave mixing for plasmonic structures,” Appl. Phys. A, vol. 123, no. 3, p. 152, 2017. <https://link.springer.com/article/10.1007%2Fs00339-016-0733-0>`_
2. TPB


Main features
=============

In addition to the standard TMM features this package also supports:

* Calculation of Gaussian beam (or any other beam) propagartion inside layered structures
* Calculation of nonlinear processes SHG/SFG/DFG

Technical features
==================

* Written in C++
* Python wrappers written in Cython
* Parallerization through OpenMP
* Use of SSE instructions for speedup