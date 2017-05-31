import pytest
import numpy as np
from LabPy import Constants
from NonlinearTMM import SecondOrderNLTMM, Material

def SpdcPowerQuantum(wlP1, wlP2, betaP1, betaP2, nF, chi2, crystalL, pwrP1, dwl, solidAngleSpdc, deltaThetaSpdc):
    wlGen = Constants.OmegaToWl(Constants.WlToOmega(wlP1) - Constants.WlToOmega(wlP2))
    betaGen = wlGen * (betaP1 / wlP1 - betaP2 / wlP2);
    
    # Omegas
    omegaP1 = Constants.WlToOmega(wlP1)
    omegaP2 = Constants.WlToOmega(wlP2)
    omegaGen = Constants.WlToOmega(wlGen)
    
    # Refractive indices
    nP1 = nF(wlP1)
    nP2 = nF(wlP2)
    nGen = nF(wlGen)
    
    # Wave vectors
    k0P1 = 2.0 * np.pi / wlP1
    k0P2 = 2.0 * np.pi / wlP2
    k0Gen = 2.0 * np.pi / wlGen
    kP1 = k0P1 * nP1
    kP2 = k0P2 * nP2
    kGen = k0Gen * nGen
    
    # Pump 1
    kxP1 = betaP1 * k0P1
    kzP1 = np.sqrt(kP1 ** 2.0 - kxP1 ** 2.0)
    
    # Pump 2 / vacuum fluctuations
    kxP2 = k0P2 * betaP2
    kzP2 = np.sqrt(kP2 ** 2.0 - kxP2 ** 2.0)
    
    # Gen
    kxGen = betaGen * k0Gen
    kzGenSqr = kGen ** 2.0 - kxGen ** 2.0
    kzGen = np.sqrt(kzGenSqr)
    
    # dk
    dkz = kzP1 - kzP2 - kzGen
    #print("wlGen, betaGen, dkz", 1e9 * wlGen, betaGen, dkz)
        
    res = dwl * solidAngleSpdc / deltaThetaSpdc * \
        Constants.hp * chi2 ** 2.0 / (8.0 * np.pi ** 4 * Constants.eps0) * \
        omegaGen ** 6 * omegaP1 * omegaP2 ** 2 / (Constants.c ** 8) * pwrP1 * \
        1.0 / (kzGen * kzP1 * kP2) * \
        np.sin(0.5 * dkz * crystalL) ** 2 / (0.5 * dkz) ** 2
        
    return res


@pytest.fixture(params = [0.1e-3, 1e-3])
def crystalD(request):
    return request.param

@pytest.fixture(params = [0.1, 1.0])
def pwrP1(request):
    return request.param

@pytest.fixture(params = [1.61681049944, 1.5])
def n1(request):
    return request.param

@pytest.fixture(params = [1.6913720951, 1.52])
def n2(request):
    return request.param

@pytest.fixture()
def tmmParams(crystalD, pwrP1, n1, n2):
    # Define params
    wlP1 = 400e-9
    wlP2 = 800e-9
    polP1 = "s"
    polP2 = "s"
    polGen = "s"
    
    deltaThetaSpdc = np.radians(0.5)
    solidAngleSpdc = 7.61543549467e-05
    deltaWlSpdc = 2.5e-9
    chi2 = 1.4761823412e-12
    
    # Define materials
    wlsCrystal = np.array([400e-9, 800.0001e-9])
    nsCrystal = np.array([n1, n2], dtype = complex)
    prism = Material(wlsCrystal, nsCrystal)
    crystal = Material(wlsCrystal, nsCrystal)
    dielectric = Material(wlsCrystal, nsCrystal)
    crystal.chi2.Update(chi222 = chi2, chi111 = chi2, chi333 = chi2)
    
    # Init SecondOrderNLTMM
    tmm = SecondOrderNLTMM(mode = "spdc", deltaWlSpdc = deltaWlSpdc, \
                           solidAngleSpdc = solidAngleSpdc, deltaThetaSpdc = deltaThetaSpdc)
    tmm.P1.SetParams(wl = wlP1, pol = polP1, I0 = pwrP1)
    tmm.P2.SetParams(wl = wlP2, pol = polP2, overrideE0 = True, E0 = 1.0)
    tmm.Gen.SetParams(pol = polGen)
    
    # Add layers
    tmm.AddLayer(float("inf"), prism)
    tmm.AddLayer(crystalD, crystal)
    tmm.AddLayer(float("inf"), dielectric)
    
    return tmm, prism, chi2
    
    
def test_SPDCIntensity(tmmParams):
    tmm, nF, chi2 = tmmParams
    betaP1 = 0.0
    betasP2 = np.linspace(0.0, 0.9, 10000)
    betasP1 = np.ones_like(betasP2) * betaP1
    pwrP1 = tmm.P1.I0
    crystalD = tmm.P1.layers[1].d
    
    # Sweep over beta = sin(th) * n_prism
    sr = tmm.Sweep("beta", betasP1, betasP2)
    quantumSpdc = SpdcPowerQuantum(tmm.P1.wl, tmm.P2.wl, betaP1, betasP2, \
        lambda wl: nF.GetN(wl).real, chi2, crystalD, pwrP1, tmm.deltaWlSpdc, tmm.solidAngleSpdc, tmm.deltaThetaSpdc)
    
    np.testing.assert_allclose(quantumSpdc, sr.Gen.It, rtol = 1e-6)

