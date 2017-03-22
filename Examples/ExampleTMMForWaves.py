import numpy as np
import pylab as plt
from NonlinearTMM import TMM, Material

def CalcSppGaussianBeam():
    # Parameters
    #---------------------------------------------------------------------------
    wl = 532e-9  # Wavelength
    pol = "p"  # Polarization
    I0 = 1.0  # Intensity of incident wave
    metalD = 50e-9  # Metal film thickness
    enhLayer = 2  # Measure enhancment in the last layer
    ths = np.radians(np.linspace(0.0, 75.0, 500))  # Angle of incidences
    xs = np.linspace(-50e-6, 50e-6, 200)  # Field calculation coordinates
    zs = np.linspace(-25e-6, 5e-6, 201)  # Field calculation coordinates
    waveType = "gaussian"  # Wave type
    pwr = 10e-3  # Beam power [W]
    w0 = 10e-6  # Beam waist size
    
    # Specify materials
    #---------------------------------------------------------------------------
    prism = Material.Static(1.5)
    ag = Material.Static(0.054007 + 3.4290j)  # Johnson & Christie @ 532nm
    dielectric = Material.Static(1.0)
    
    # Init TMM
    #---------------------------------------------------------------------------
    tmm = TMM(wl = wl, pol = pol, I0 = I0)
    tmm.AddLayer(float("inf"), prism)
    tmm.AddLayer(metalD, ag)
    tmm.AddLayer(float("inf"), dielectric)
    
    # Init wave params
    tmm.wave.SetParams(waveType = waveType, w0 = w0, pwr = pwr, \
                       dynamicMaxX = False, maxX = xs[-1])
    
    # Solve
    #---------------------------------------------------------------------------
    
    # Calculate reflection, transmission and field enhancement
    betas = np.sin(ths) * prism.GetN(wl).real
    sweepRes = tmm.WaveSweep("beta", betas, outEnh = True, layerNr = enhLayer) 
    
    # Calculate fields at the reflection dip (excitation of SPPs)
    betaMaxEnh = betas[np.argmax(sweepRes.enh)]
    tmm.Solve(beta = betaMaxEnh)
    fields2D = tmm.WaveGetFields2D(zs, xs)
    
    # Ploting
    #---------------------------------------------------------------------------
    plt.figure()
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan = 2)
    thMaxEnh = np.arcsin(betaMaxEnh / prism.GetN(wl).real)
    
    # Reflection / transmission
    ax1.plot(np.degrees(ths), 1e3 * sweepRes.Pi, label = r"$P_i$")
    ax1.plot(np.degrees(ths), 1e3 * sweepRes.Pr, label = r"$P_r$")
    ax1.plot(np.degrees(ths), 1e3 * sweepRes.Pt, label = r"$P_t$")
    ax1.axvline(np.degrees(thMaxEnh), ls = "--", color = "red", lw = 1.0)
    ax1.set_xlabel(r"$\theta$ ($\degree$)")
    ax1.set_ylabel(r"Power (mW)")
    ax1.legend()
    
    # Field enhancement
    ax2.plot(np.degrees(ths), sweepRes.enh)
    ax2.axvline(np.degrees(thMaxEnh), ls = "--", color = "red", lw = 1.0)
    ax2.set_xlabel(r"$\theta$ ($\degree$)")
    ax2.set_ylabel(r"Field enhancement")
    
    # Fields 2D
    cm = ax3.pcolormesh(1e6 * zs, 1e6 * xs, 1e-3 * fields2D.EN.real.T, vmax = 5e1)
    ax3.set_xlabel(r"z (μm)")
    ax3.set_ylabel(r"x (μm)")
    plt.colorbar(cm, label = r"$‖E‖$ (kV/m)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    CalcSppGaussianBeam()
