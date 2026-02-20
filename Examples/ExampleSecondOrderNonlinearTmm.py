from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from NonlinearTMM import Material, SecondOrderNLTMM


def CalcSHG() -> None:
    # Parameters
    # ---------------------------------------------------------------------------
    wl = 1000e-9  # Pump wavelength
    pol = "s"  # Polarization
    I0 = 1.0  # Intensity of incident pump wave
    crystalD = 1000e-6  # Crystal thickness
    betas = np.linspace(0.0, 0.99, 10000)  # Sweep range for beta

    # Define materials
    # ---------------------------------------------------------------------------
    wlsCrystal = np.array([400e-9, 1100e-9])
    nsCrystal = np.array([1.54, 1.53], dtype=complex)
    prism = Material.Static(1.0)
    crystal = Material(wlsCrystal, nsCrystal)
    crystal.chi2.Update(d22=1e-12)
    dielectric = Material.Static(1.0)

    # Init SecondOrderNLTMM
    # ---------------------------------------------------------------------------
    tmm = SecondOrderNLTMM()
    tmm.P1.SetParams(wl=wl, pol=pol, beta=0.2, I0=I0)
    tmm.P2.SetParams(wl=wl, pol=pol, beta=0.2, I0=I0)
    tmm.Gen.SetParams(pol=pol)

    # Add layers
    tmm.AddLayer(math.inf, prism)
    tmm.AddLayer(crystalD, crystal)
    tmm.AddLayer(math.inf, dielectric)

    # Beta sweep
    # ---------------------------------------------------------------------------
    sr = tmm.Sweep("beta", betas, betas, outP1=True, outGen=True)

    # Crystal thickness sweep at normal incidence (beta = 0)
    # ---------------------------------------------------------------------------
    thicknesses = np.linspace(10e-6, 2000e-6, 200)
    shg_t = np.empty(len(thicknesses))
    for i, d in enumerate(thicknesses):
        tmm2 = SecondOrderNLTMM()
        tmm2.P1.SetParams(wl=wl, pol=pol, beta=0.0, I0=I0)
        tmm2.P2.SetParams(wl=wl, pol=pol, beta=0.0, I0=I0)
        tmm2.Gen.SetParams(pol=pol)
        tmm2.AddLayer(math.inf, prism)
        tmm2.AddLayer(d, crystal)
        tmm2.AddLayer(math.inf, dielectric)
        tmm2.Solve()
        intensities = tmm2.GetIntensities()
        shg_t[i] = intensities.Gen.T

    # Plot results
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))

    # Left: Schematic of the setup
    ax = axes[0]
    ax.set_xlim(-1, 5)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Setup")

    # Draw layers
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((-0.5, -1.5), 1.5, 3, fc="#ddeeff", ec="k", lw=0.8))
    ax.add_patch(Rectangle((1, -1.5), 2, 3, fc="#ffe0cc", ec="k", lw=1.2))
    ax.add_patch(Rectangle((3, -1.5), 1.5, 3, fc="#ddeeff", ec="k", lw=0.8))
    ax.text(0.25, -1.8, "air", ha="center", fontsize=8)
    ax.text(2.0, -1.8, r"$\chi^{(2)}$ crystal", ha="center", fontsize=8)
    ax.text(3.75, -1.8, "air", ha="center", fontsize=8)

    # Pump arrow
    ax.annotate(
        "",
        xy=(0.9, 0.3),
        xytext=(-0.6, 0.3),
        arrowprops=dict(arrowstyle="-|>", color="C0", lw=2),
    )
    ax.text(-0.5, 0.6, r"$\omega$ pump", fontsize=7, color="C0")

    # SHG arrows (reflected + transmitted)
    ax.annotate(
        "",
        xy=(-0.6, -0.3),
        xytext=(0.9, -0.3),
        arrowprops=dict(arrowstyle="-|>", color="C3", lw=1.5, ls="--"),
    )
    ax.text(-0.5, -0.7, r"$2\omega$ R", fontsize=7, color="C3")

    ax.annotate(
        "",
        xy=(4.6, -0.3),
        xytext=(3.1, -0.3),
        arrowprops=dict(arrowstyle="-|>", color="C3", lw=2),
    )
    ax.text(3.7, -0.7, r"$2\omega$ T", fontsize=7, color="C3")

    # Middle: SHG R, T vs beta
    ax = axes[1]
    ax.plot(betas, sr.Gen.Ir, label="R")
    ax.plot(betas, sr.Gen.It, label="T")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"Intensity ($W/m^{2}$)")
    ax.set_title(r"SHG intensity vs $\beta$")
    ax.legend()

    # Right: SHG T vs crystal thickness
    ax = axes[2]
    ax.plot(thicknesses * 1e6, shg_t)
    ax.set_xlabel(r"Crystal thickness ($\mu m$)")
    ax.set_ylabel(r"SHG transmitted ($W/m^{2}$)")
    ax.set_title(r"Thickness dependence ($\beta$ = 0)")

    fig.tight_layout()
    fig.savefig("docs/images/SecondOrderNLTMM-example.png", dpi=100)
    plt.show()


if __name__ == "__main__":
    CalcSHG()
