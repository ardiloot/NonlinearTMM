from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from NonlinearTMM import TMM, Material


def CalcSpp():
    # Parameters
    # ---------------------------------------------------------------------------
    wl = 532e-9  # Wavelength
    pol = "p"  # Polarization
    I0 = 1.0  # Intensity of incident wave
    metalD = 50e-9  # Metal film thickness
    enhLayer = 2  # Measure enhancment in the last layer
    ths = np.radians(np.linspace(0.0, 80.0, 500))  # Angle of incidences
    xs = np.linspace(-2e-6, 2e-6, 200)  # Field calculation coordinates
    zs = np.linspace(-2e-6, 2e-6, 201)  # Field calculation coordinates

    # Specify materials
    # ---------------------------------------------------------------------------
    prism = Material.Static(1.5)
    ag = Material.Static(0.054007 + 3.4290j)  # Johnson & Christie @ 532nm
    dielectric = Material.Static(1.0)

    # Init TMM
    # ---------------------------------------------------------------------------
    tmm = TMM(wl=wl, pol=pol, I0=I0)
    tmm.AddLayer(math.inf, prism)
    tmm.AddLayer(metalD, ag)
    tmm.AddLayer(math.inf, dielectric)

    # Solve
    # ---------------------------------------------------------------------------

    # Calculate reflection, transmission and field enhancement
    betas = np.sin(ths) * prism.GetN(wl).real
    sweepRes = tmm.Sweep("beta", betas, outEnh=True, layerNr=enhLayer)

    # Calculate fields at the reflection dip (excitation of SPPs)
    betaMaxEnh = betas[np.argmax(sweepRes.enh)]
    tmm.Solve(beta=betaMaxEnh)

    # Calculate 1D fields
    fields1D = tmm.GetFields(zs)

    # Calculate 2D fields
    fields2D = tmm.GetFields2D(zs, xs)

    # Ploting
    # ---------------------------------------------------------------------------
    plt.figure()
    thMaxEnh = np.arcsin(betaMaxEnh / prism.GetN(wl).real)

    # Reflection / transmission
    plt.subplot(221)
    plt.plot(np.degrees(ths), sweepRes.Ir, label="R")
    plt.plot(np.degrees(ths), sweepRes.It, label="T")
    plt.axvline(np.degrees(thMaxEnh), ls="--", color="red", lw=1.0)
    plt.xlabel(r"$\theta$ ($\degree$)")
    plt.ylabel(r"Intensity (a.u)")
    plt.legend()

    # Field enhancement
    plt.subplot(222)
    plt.plot(np.degrees(ths), sweepRes.enh)
    plt.axvline(np.degrees(thMaxEnh), ls="--", color="red", lw=1.0)
    plt.xlabel(r"$\theta$ ($\degree$)")
    plt.ylabel(r"Field enhancement")

    # Fields 1D
    plt.subplot(223)
    plt.plot(1e6 * zs, fields1D.E[:, 0].real, label=r"$E_x$")
    plt.plot(1e6 * zs, fields1D.E[:, 2].real, label=r"$E_z$")
    plt.plot(1e6 * zs, np.linalg.norm(fields1D.E, axis=1), label=r"‖E‖")
    plt.xlabel(r"z (μm)")
    plt.ylabel(r"(V/m)")
    plt.legend()

    # Fields 2D
    plt.subplot(224)
    plt.pcolormesh(1e6 * zs, 1e6 * xs, fields2D.Ez.real.T, rasterized=True)
    plt.xlabel(r"z (μm)")
    plt.ylabel(r"x (μm)")
    plt.colorbar(label=r"$E_z$ (V/m)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    CalcSpp()
