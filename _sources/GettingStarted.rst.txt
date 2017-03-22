.. py:currentmodule:: NonlinearTMM

Getting started
###############

Installation
************

Installation of NonlinearTMM package is possible through pip or from source
code.

Requirements:

* `Python >3.0 <https://www.python.org/>`_
* C++ compiler
* `Cython <http://cython.org/>`_ for wrapping C++ code to Python
* `Numpy <http://www.numpy.org/>`_ for general arrays
* `Eigency <https://github.com/wouterboomsma/eigency>`_ for Eigen and Numpy array conversion
* `Matplotlib <http://matplotlib.org/>`_ for plotting in examples
* `Pytest <http://doc.pytest.org/>`_ for testing

Tested with Python 3.6, Windows 10 x64, Visual studio C++ 2015.

Dependencies:

* C++ code depends on `Eigen library <http://eigen.tuxfamily.org/>`_ (already included in package) 

Installation through pip is done like:

.. code-block:: bash

	pip install NonlinearTMM

Alternatively, it is possible to install the package form the source code by commands

.. code-block:: bash

	python setup.py install

or

.. code-block:: bash

	pip intall .
	
in the source code folder.


Package structure
*****************

The package has three main classes:

* :class:`Material`
* :class:`TMM`
* :class:`SecondOrderNLTMM`

Class :class:`Material` is responsible to represent the properties of optical
material. Mainly wavelength dependent refractive indices and second-order
suceptibility tensor for nonlinear processes.

Class :class:`TMM` (alias of :class:`NonlinearTMM`) is has all the standard TMM features:

* Both p- and s-polarization
* Arbritarty angle of incidence
* Calculation of reflection, transmission and absorption of plane waves (:any:`GetIntensities <NonlinearTMM.GetIntensities>` and :any:`GetAbsorbedIntensity <NonlinearTMM.GetAbsorbedIntensity>`)
* Calculaion of electric and magnetic fields inside structure (:any:`GetFields <NonlinearTMM.GetFields>` and :any:`GetFields2D <NonlinearTMM.GetFields2D>`)
* Calculation of field enhancement (:any:`GetEnhancement <NonlinearTMM.GetEnhancement>`)
* Sweep over any parameter (:any:`Sweep <NonlinearTMM.Sweep>`)

In addition to those standard features, the class has similar functionality to
work with waves with arbritarty profile (e.g. Gaussian beam). The configuration
of the beam is done through attribute :any:`wave <NonlinearTMM.wave>` (see class :any:`_Wave`).
The interface for the calculations with arbritarty beams is similar to standard TMM:

* Calculation of reflection, transmission and absorption of beams (:any:`WaveGetPowerFlows <NonlinearTMM.WaveGetPowerFlows>`)
* Calculaion of electric and magnetic fields inside structure (:any:`WaveGetFields2D <NonlinearTMM.WaveGetFields2D>`)
* Calculation of field enhancement (:any:`WaveGetEnhancement <NonlinearTMM.WaveGetEnhancement>`)
* Sweep over any parameter (:any:`WaveSweep <NonlinearTMM.WaveSweep>`)

Finally, :class:`SecondOrderNLTMM` class ic capable of calculating second-order
nonlinear processes like second-harmonic generation, sum-frequency generation and
difference frequency generation. This has similar interface as :any:`TMM` - it
supports both the plane waves and beams. 

Standard TMM
************

Plane waves example
===================

As an example three layer structure consisting of a prism (z < 0), 50 nm thick silver
film and air is studied. Such kind of structure supports surface plasmon
resonance (SPP) if excited by p-polarized light and is named Kretschmann
configuration. The example code is shown bellow and could be divided into
following steps:

1. Specifying materials refractive indices.
2. Initializing :class:`TMM`, setting params and adding layers.
3. By using :any:`Sweep <NonlinearTMM.Sweep>` calculate the dependence of reflection, transmission and enhancment factor on the angle of incidence.
4. Find the plasmonic resonance by maximum enhancment.
5. Calculate 1D fields at plasmonic resonance by :any:`GetFields <NonlinearTMM.GetFields>`.
6. Calculate 2D fields at plasmonic resonance by :any:`GetFields2D <NonlinearTMM.GetFields2D>`.
7. Plot all results
 
.. literalinclude:: ../Examples/ExampleTMM.py

The results of the calculations are shown bellow. Indeed there is a sharrp dip 
in the reflection (R) near the angle of incidence ca 44 degrees. At the same angle
the field enhancement factor is maximum and is more than 12 times. In the second
the results of the fields calculations at plasmonic resonance is presented. Indeed,
surface wave on the silver-air interface is excited and characteristic pattern of
fields for SPP is visible.

.. image:: images/TMM-example.png

Gaussian wave example
=====================

Previous example was entirely about standard TMM. Now, the calculations are
extended to the beams, in this case Gaussian beam. The steps of the calculations
remain the same, except :class:`_Wave` parameters must be set (:class:`TMM` has
attribute :any:`TMM.wave`). Gaussian beam power is set to 10 mW and waist size
to 10 μm.

.. literalinclude:: ../Examples/ExampleTMMForWaves.py

The results of those calculations are bellow. Despite the fact, that the structure
is the same, the dip in the reflection is different. The reason for this behaviour
is that as the resonances of SPPs are narrow, they also require well collimated
beam to excite them. Also field enhancment is ca 3 times lower, as expected. On
the right side, the electrical field norm is shown. It is clearly visible, that
Gaussian beam is incident form the left, and it gets reflected from the metal film (z = 0).
Part of the energy is transmitted to excite SPPs at the metal-air interface. The
excited SPPs are propagating on the metal film and are absorbe after ca 20 μm of
propagation. 

.. image:: images/TMMForWaves-example.png

Second-order nonlinear TMM
**************************

Plane waves example
===================

Will be added in near future.

Gaussian wave example
=====================

Will be added in near future.

