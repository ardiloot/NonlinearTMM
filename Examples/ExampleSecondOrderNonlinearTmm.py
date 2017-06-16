import numpy as np
import pylab as plt
from NonlinearTMM import SecondOrderNLTMM, Material

if __name__ == "__main__":
    # Define params
    wlP1 = 1000e-9
    wlP2 = 1000e-9
    polP1 = "s"
    polP2 = "s"
    polGen = "s"
    I0P1 = 1.0
    I0P2 = I0P1
    betas = np.linspace(0.0, 0.99, 10000)
    crystalD = 1000e-6
    
    # Define materials
    wlsCrystal = np.array([400e-9, 1100e-9])
    nsCrystal = np.array([1.54, 1.53], dtype = complex)
    prism = Material.Static(1.0)
    crystal = Material(wlsCrystal, nsCrystal)
    dielectric = Material.Static(1.0)
    crystal.chi2.Update(d22 = 1e-12)
    
    # Init SecondOrderNLTMM
    tmm = SecondOrderNLTMM()
    tmm.P1.SetParams(wl = 1000e-9, pol = "s", beta = 0.2, I0 = 1.0)
    tmm.P2.SetParams(wl = 1000e-9, pol = "s", beta = 0.2, I0 = 1.0)
    tmm.Gen.SetParams(pol = "s")
    
    # Add layers
    tmm.AddLayer(float("inf"), prism)
    tmm.AddLayer(crystalD, crystal)
    tmm.AddLayer(float("inf"), dielectric)
    
    # Sweep over beta = sin(th) * n_prism
    sr = tmm.Sweep("beta", betas, betas)
    
    # Plot generated reflection and transmission
    plt.title("SHG generation from crystal (d = %.0f $\mu m$)" % (1e6 * crystalD))
    plt.plot(betas, sr.Gen.Ir, label = "R")
    plt.plot(betas, sr.Gen.It, label = "T")
    plt.legend()
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"($W / m^{2}$)")
        
    plt.show()
    
    