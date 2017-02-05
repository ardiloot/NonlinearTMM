import numpy as np
import pylab as plt
from NonlinearTMM import NonlinearTMM, Material

def CalcSpp():
    # Parameters
    wl = 532e-9     # Wavelength
    pol = "p"       # Polarizations
    I0 = 1.0        # Intensity of incident wave
    metalD = 50e-9  # Metal film thickness
    
    # Specify materials
    prism = Material.Static(1.5)
    ag = Material.Static(0.054007 + 3.4290j) # Johnson & Christie @ 532nm
    dielectric = Material.Static(1.0)
    
    # Init TMM
    tmm = NonlinearTMM(wl = wl, pol = pol, I0 = I0)
    tmm.AddLayer(float("inf"), prism)
    tmm.AddLayer(metalD, ag)
    tmm.AddLayer(float("inf"), dielectric)
    
    # Specify angle of incidences
    ths = np.radians(np.linspace(0.0, 80.0, 500))
    betas = np.sin(ths) * prism.GetN(wl).real
    
    # Calculate reflection/transmission
    sweepRes = tmm.Sweep("beta", betas) 
    
    # Plot
    plt.figure()
    plt.plot(np.degrees(ths), sweepRes.R, label = "R")
    plt.plot(np.degrees(ths), sweepRes.T, label = "T")
    plt.xlabel(r"$\theta$ ($\degree$)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    CalcSpp()