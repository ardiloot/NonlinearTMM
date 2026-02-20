.. py:currentmodule:: NonlinearTMM

Getting started
###############

Installation
************

Installation of the NonlinearTMM package is possible through pip or from source
code.

Requirements:

* `Python >= 3.10 <https://www.python.org/>`_

Dependencies:

* C++ code depends on `Eigen library <https://eigen.tuxfamily.org/>`_ (already included in package)

Installation through pip is done like:

.. code-block:: bash

	pip install NonlinearTMM

Alternatively, it is possible to install the package from the source code by

.. code-block:: bash

	pip install .

in the source code folder.


Package structure
*****************

The package has three main classes:

* :class:`Material`
* :class:`TMM`
* :class:`SecondOrderNLTMM`

Class :class:`Material` is responsible for representing the properties of optical
material, mainly wavelength-dependent refractive indices and second-order
susceptibility tensor for nonlinear processes.

Class :class:`TMM` (alias of :class:`NonlinearTMM`) has all the standard TMM features:

* Both p- and s-polarization
* Arbitrary angle of incidence
* Calculation of reflection, transmission and absorption of plane waves (:meth:`GetIntensities <NonlinearTMM.GetIntensities>` and :meth:`GetAbsorbedIntensity <NonlinearTMM.GetAbsorbedIntensity>`)
* Calculation of electric and magnetic fields inside structure (:meth:`GetFields <NonlinearTMM.GetFields>` and :meth:`GetFields2D <NonlinearTMM.GetFields2D>`)
* Calculation of field enhancement (:meth:`GetEnhancement <NonlinearTMM.GetEnhancement>`)
* Sweep over any parameter (:meth:`Sweep <NonlinearTMM.Sweep>`)

In addition to those standard features, the class has similar functionality to
work with waves with arbitrary profile (e.g. Gaussian beam). The configuration
of the beam is done through attribute :attr:`wave <NonlinearTMM.wave>` (see class :class:`_Wave`).
The interface for the calculations with arbitrary beams is similar to standard TMM:

* Calculation of reflection, transmission and absorption of beams (:meth:`WaveGetPowerFlows <NonlinearTMM.WaveGetPowerFlows>`)
* Calculation of electric and magnetic fields inside structure (:meth:`WaveGetFields2D <NonlinearTMM.WaveGetFields2D>`)
* Calculation of field enhancement (:meth:`WaveGetEnhancement <NonlinearTMM.WaveGetEnhancement>`)
* Sweep over any parameter (:meth:`WaveSweep <NonlinearTMM.WaveSweep>`)

Finally, :class:`SecondOrderNLTMM` class is capable of calculating second-order
nonlinear processes like second-harmonic generation, sum-frequency generation and
difference frequency generation. This has a similar interface to :class:`TMM` - it
supports both plane waves and beams.

Standard TMM
************

Plane waves example
===================

As an example, a three-layer structure consisting of a prism (z < 0), a 50-nm-thick silver
film and air is studied. Such a structure supports surface plasmon
resonance (SPR) if excited by p-polarized light and is known as the Kretschmann
configuration. The example code is shown below and could be divided into
the following steps:

1. Specifying material refractive indices.
2. Initializing :class:`TMM`, setting params and adding layers.
3. By using :meth:`Sweep <NonlinearTMM.Sweep>` calculate the dependence of reflection, transmission and enhancement factor on the angle of incidence.
4. Find the plasmonic resonance from the maximum enhancement.
5. Calculate 1D fields at plasmonic resonance by :meth:`GetFields <NonlinearTMM.GetFields>`.
6. Calculate 2D fields at plasmonic resonance by :meth:`GetFields2D <NonlinearTMM.GetFields2D>`.
7. Plot all results

.. literalinclude:: ../Examples/ExampleTMM.py

The results of the calculations are shown below. Indeed, there is a sharp dip
in the reflection (R) near the angle of incidence of approximately 44 degrees. At the same angle,
the field enhancement factor is at its maximum and is more than 12 times the incident field. In the lower
panels, the results of the field calculations at plasmonic resonance are presented.
Indeed, a surface wave on the silver-air interface is excited and the characteristic
pattern of fields for SPP is visible.

.. image:: images/TMM-example.png

Gaussian wave example
=====================

The previous example was entirely about standard TMM. Now, the calculations are
extended to beams, in this case a Gaussian beam. The steps of the calculations
remain the same, except :class:`_Wave` parameters must be set (:class:`TMM` has
attribute :attr:`TMM.wave`). The Gaussian beam power is set to 10 mW and the waist size
to 10 μm.

.. literalinclude:: ../Examples/ExampleTMMForWaves.py

The results of those calculations are below. Despite the fact that the structure
is the same, the dip in the reflection is different. The reason for this behaviour
is that as the resonances of SPPs are narrow, they also require a well-collimated
beam to excite them. Also, the field enhancement is approximately 3 times lower, as expected. On
the right side, the electric field norm is shown. It is clearly visible that
a Gaussian beam is incident from the left, and it gets reflected from the metal film (z = 0).
Part of the energy is transmitted to excite SPPs at the metal-air interface. The
excited SPPs are propagating on the metal film and are absorbed after approximately 20 μm of
propagation.

.. image:: images/TMMForWaves-example.png

Second-order nonlinear TMM
**************************

Plane waves example
===================

As an example, second-harmonic generation (SHG) in a nonlinear crystal is
calculated. The example code is shown below.

.. literalinclude:: ../Examples/ExampleSecondOrderNonlinearTmm.py

The results show the reflected and transmitted SHG intensity as a function
of the propagation parameter β. Two s-polarized pump beams at 1000 nm
generate a second-harmonic signal at 500 nm in a 1 mm nonlinear crystal.

.. image:: images/SecondOrderNLTMM-example.png
