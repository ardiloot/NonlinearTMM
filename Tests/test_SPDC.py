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

@pytest.fixture(params = [0.0, 0.3])
def betaP1(request):
    return request.param

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

@pytest.fixture(params = ["planewave", "gaussian"])
def waveP1Type(request):
    return request.param

@pytest.fixture()
def tmmParams(crystalD, pwrP1, n1, n2):
    # Define params
    wlP1 = 450e-9
    wlP2 = 800e-9
    polP1 = "s"
    polP2 = "s"
    polGen = "s"
    w0 = 1e-3
    Ly = 1e-3
    I0 = pwrP1  / (w0 * Ly)
    
    deltaThetaSpdc = np.radians(0.5)
    solidAngleSpdc = 7.61543549467e-05
    deltaWlSpdc = 2.5e-9
    chi2 = 1.4761823412e-12
    
    # Define materials
    wlsCrystal = np.array([100e-9, 1200.0001e-9])
    nsCrystal = np.array([n1, n2], dtype = complex)
    prism = Material(wlsCrystal, nsCrystal)
    crystal = Material(wlsCrystal, nsCrystal)
    dielectric = Material(wlsCrystal, nsCrystal)
    crystal.chi2.Update(chi222 = chi2, chi111 = chi2, chi333 = chi2)
    
    # Init SecondOrderNLTMM
    tmm = SecondOrderNLTMM(mode = "spdc", deltaWlSpdc = deltaWlSpdc, \
                           solidAngleSpdc = solidAngleSpdc, deltaThetaSpdc = deltaThetaSpdc)
    tmm.P1.SetParams(wl = wlP1, pol = polP1, I0 = I0)
    tmm.P2.SetParams(wl = wlP2, pol = polP2, overrideE0 = True, E0 = 1.0)
    tmm.Gen.SetParams(pol = polGen)
    
    # Add layers
    tmm.AddLayer(float("inf"), prism)
    tmm.AddLayer(crystalD, crystal)
    tmm.AddLayer(float("inf"), dielectric)
    
    return tmm, prism, chi2, w0, Ly
    
    
def test_SPDCIntensity(tmmParams, betaP1):
    tmm, nF, chi2, w0, Ly = tmmParams
    betasP2 = np.linspace(0.0, 0.9, 10000)
    betasP1 = np.ones_like(betasP2) * betaP1
    pwrP1 = tmm.P1.I0
    crystalD = tmm.P1.layers[1].d
    
    # Sweep over beta = sin(th) * n_prism
    sr = tmm.Sweep("beta", betasP1, betasP2)
    quantumSpdc = SpdcPowerQuantum(tmm.P1.wl, tmm.P2.wl, betaP1, betasP2, \
        lambda wl: nF.GetN(wl).real, chi2, crystalD, pwrP1, tmm.deltaWlSpdc, tmm.solidAngleSpdc, tmm.deltaThetaSpdc)
    coef = 1.0 / np.cos(np.arcsin(betasP1 / nF.GetN(tmm.P1.wl).real)) 
    
    np.testing.assert_allclose(quantumSpdc, sr.Gen.It * coef, rtol = 1e-6)

def test_SPDCPwrs(tmmParams, waveP1Type, betaP1):
    tmm, nF, chi2, w0, Ly = tmmParams 
    betasP2Test = np.linspace(0.0, 0.9, 10000)
    betasP1Test = np.ones_like(betasP2Test) * betaP1
    
    # Waves
    pwrP1 = tmm.P1.I0 * w0 * Ly
    waveP1Params = {"waveType": waveP1Type, "pwr": pwrP1, "w0": w0, "Ly": Ly, "dynamicMaxXCoef": 10, "nPointsInteg": 100, "maxPhi": np.radians(10.0)}
    waveP2Params = {"waveType": "spdc", "w0": w0, "Ly": Ly, "overrideE0": True, "nPointsInteg": 100}
    tmm.P1.wave.SetParams(**waveP1Params)
    tmm.P2.wave.SetParams(**waveP2Params)

    # Find maximum signal
    srTest = tmm.Sweep("beta", betasP1Test, betasP2Test)
    index = np.argmax(srTest.Gen.It)
    betaP2 = betasP2Test[index]
    
    # Manual integration
    # --------------------------------------------------------------------------
    
    # Find betaP2 integration range
    wlGen = srTest.wlsGen[0]
    n0Gen = nF.GetN(wlGen).real
    maxBetaGen = srTest.betasGen[index]
    kz0Gen = (2.0 * np.pi / wlGen) * np.sqrt(n0Gen ** 2 - maxBetaGen ** 2)
    deltaKxP2 = kz0Gen / (2.0 * n0Gen) * tmm.deltaThetaSpdc
    deltaBetaP2 = deltaKxP2 / (2.0 * np.pi / tmm.P2.wl)
    
    # Integrate
    betasP2Int = np.linspace(betaP2 - deltaBetaP2, betaP2 + deltaBetaP2, 100)
    betasP1Int = np.ones_like(betasP2Int) * betaP1
    kxsP2Int = betasP2Int * (2.0 * np.pi / tmm.P2.wl)
    srInt = tmm.Sweep("beta", betasP1Int, betasP2Int)
    
    # Int power
    pwrSPDC = np.trapz(srInt.Gen.It, kxsP2Int)
    
    # Wave integration
    # --------------------------------------------------------------------------
    
    tmm.P1.beta = betaP1
    tmm.P2.beta = betaP2
    pwrSpdcW = tmm.WaveGetPowerFlows(2)[0]
    beamArea = tmm.P1.wave.beamArea
    
    # Testing
    # --------------------------------------------------------------------------
    np.testing.assert_allclose(beamArea * pwrSPDC, pwrSpdcW, rtol = 1e-2)
    
    
    

