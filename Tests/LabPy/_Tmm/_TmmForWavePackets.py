import numpy as np
from LabPy import Core, Constants
from time import time

__all__ = ["PlaneWave", \
           "GaussianWave",
           "GaussianWaveFFT",
           "TukeyWaveFFT",
           "TmmForWaves",
           "SecondOrderNLTmmForWaves"]

#===============================================================================
# PlaneWave
#===============================================================================

class PlaneWave(Core.ParamsBaseClass):
    
    def __init__(self, params = [], **kwargs):
        paramsThis = ["pwr", "overrideE0", "w0", "n", "Ly"]
        self.overrideE0 = None #E0 specified in vacuum, not in first layer
        self.Ly = None
        super(PlaneWave, self).__init__(params + paramsThis, **kwargs)

    def Solve(self, wl, th0, **kwargs):
        self.wl = wl
        self.th0 = th0
        self.SetParams(**kwargs)
        
        self._Precalc()
        self._Solve()

    def _Precalc(self): 
        if self.overrideE0 is None:
            self.I0 = self.pwr / (self.Ly * self.w0) 
            self.E0 = np.sqrt(2.0 * Constants.mu0 * Constants.c * self.I0) 
            # E0 in vacuum, not in 1st layer
            print("E0 vacuum", self.E0)
        else:
            self.E0 = self.overrideE0
            
        self.k0 = 2.0 * np.pi / self.wl
        self.k = self.k0 * self.n(self.wl).real
        print("_Precalc", self.E0)
        
    def _Solve(self):
        self.phis = np.array([0.0])
        self.kxs = np.array([self.k * np.sin(self.th0)])
        self.kzs = np.array([self.k * np.cos(self.th0)])
        self.expansionCoefsPhi = np.array([self.E0])
        self.expansionCoefsKx = np.array([self.E0])
        self.betas = (self.kxs / self.k0).real
   
       
#===============================================================================
# GaussianWave
#===============================================================================

class GaussianWave(PlaneWave):
    def __init__(self, params = [], **kwargs):
        paramsThis = ["integCriteria", "nPointsInteg"]
        self.nPointsInteg = 30
        self.integCriteria = 1e-3
        super(GaussianWave, self).__init__(params + paramsThis, **kwargs)
    
    def _Solve(self):
        self._phiLim  = np.arcsin(2.0 / self.w0 * np.sqrt(-np.log(self.integCriteria)) / self.k)
        self.phis = np.linspace(-self._phiLim, self._phiLim, self.nPointsInteg)

        if np.max(abs(self.phis + self.th0)) > np.pi / 2.0:
            raise ValueError("Gaussian wave requires backward propagating waves!")
        
        kzPs, kxPs = np.cos(self.phis) * self.k, np.sin(self.phis) * self.k
        self.kxs = kxPs * np.cos(self.th0) + kzPs * np.sin(self.th0)
        self.kzs = -kxPs * np.sin(self.th0) + kzPs * np.cos(self.th0)
        profileSpectrum = self.E0 * 1.0 / 2.0 / np.sqrt(np.pi) * self.w0 * \
            np.exp(-(kxPs) ** 2.0 * self.w0 ** 2.0 / 4.0)
        self.expansionCoefsPhi = profileSpectrum * np.cos(self.phis) * self.k
        self.betas = (self.kxs / self.k0).real

#===============================================================================
# WaveFFT
#===============================================================================

class WaveFFT(PlaneWave):
    def __init__(self, params = [], **kwargs):
        paramsThis = ["nPointsInteg", "maxPhi", "integCriteria"]
        self.nPointsInteg = 30
        self.maxPhi = np.radians(0.5)
        self.integCriteria = 1e-3
        PlaneWave.__init__(self, params + paramsThis, **kwargs)
    
    def _FieldProfile(self, xs):
        raise NotImplementedError()
        
    def _Solve(self, maxPhiForce = None):
        maxKxp = np.sin(self.maxPhi if maxPhiForce is None else maxPhiForce) * self.k
        dx = np.pi / maxKxp 
        xs = np.arange(-0.5 * dx * self.nPointsInteg, 0.5 * dx * self.nPointsInteg, dx)
        
        fieldProfile = self._FieldProfile(xs)
        fieldProfileSpectrum = (xs[1] - xs[0]) / (2.0 * np.pi) * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(fieldProfile)))
        kxPs = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(fieldProfile), xs[1] - xs[0]))
        self.phis = np.arcsin(kxPs / self.k)

        if maxPhiForce is None and self.integCriteria is not None:
            # It is first iteration, need to check if it is possible to reduce the range
            # by using the integCriteria
            cumSpectra = np.cumsum(abs(fieldProfileSpectrum))
            cumSpectra /= np.max(cumSpectra)
            index = np.argmax(cumSpectra > self.integCriteria)
            
            if index > 0:
                newMaxPhi = abs(self.phis[max(index - 5, 0)])
                print("newMaxPhi", np.degrees(newMaxPhi), cumSpectra[index])
                self._Solve(newMaxPhi)
                return

        if np.max(abs(self.phis + self.th0)) > np.pi / 2.0:
            raise ValueError("Requires backward propagating waves!")
        
        kzPs = np.cos(self.phis) * self.k
        self.kxs = kxPs * np.cos(self.th0) + kzPs * np.sin(self.th0)
        self.kzs = -kxPs * np.sin(self.th0) + kzPs * np.cos(self.th0)
        
        self.expansionCoefsPhi = fieldProfileSpectrum * np.cos(self.phis) * self.k
        self.expansionCoefsKx = fieldProfileSpectrum / np.cos(self.th0)
        self.betas = (self.kxs / self.k0).real
    
#===============================================================================
# GaussianWaveFFT
#===============================================================================

class GaussianWaveFFT(WaveFFT):
    def __init__(self, params = [], **kwargs):
        paramsThis = []
        WaveFFT.__init__(self, params + paramsThis, **kwargs)
    
    def _FieldProfile(self, xs):
        res = self.E0 * np.exp(- xs ** 2.0 / self.w0 ** 2.0)
        return res

#===============================================================================
# TukeyWaveFFT
#===============================================================================

class TukeyWaveFFT(WaveFFT):
    def __init__(self, params = [], **kwargs):
        paramsThis = ["a"]
        self.a = 0.5
        WaveFFT.__init__(self, params + paramsThis, **kwargs)
    
    @staticmethod
    def TukeyFunc(xs, w0, a):
        lowering = 0.5 * np.cos(np.pi / (w0 * (1.0 - a)) * (xs - 0.5 * a * w0)) + 0.5
        raising = 0.5 * np.cos(np.pi / (w0 * (1.0 - a)) * (xs + 0.5 * (2.0 - a) * w0) + np.pi) + 0.5
        
        res = np.ones_like(xs, dtype = float)
        res = np.where(xs > 0.5 * a * w0, lowering, res)
        res = np.where(xs < -0.5 * a * w0, raising, res)
        res[xs > 0.5 * w0 * (2.0 - a)] = 0.0
        res[xs < -0.5 * w0 * (2.0 - a)] = 0.0
        return res
    
    def _FieldProfile(self, xs):
        res = self.E0 * self.TukeyFunc(xs, self.w0, self.a)
        return res
        
#===============================================================================
# WavesIntegrator1D
#===============================================================================

def WavesIntegrator1D(wave, fieldsFunc):
    resE, resH = None, None
    lastE, lastH = None, None
    dPhis = wave.phis[1:] - wave.phis[:-1] 

    for i in range(len(wave.betas)):
        fieldsE, fieldsH = fieldsFunc(wave, i)
        
        if len(wave.betas) == 1:
            return fieldsE, fieldsH
        
        if i == 0:
            resE = np.zeros_like(fieldsE)
            resH = np.zeros_like(fieldsH)    
        else:
            resE += dPhis[i - 1] * (fieldsE + lastE) / 2.0
            resH += dPhis[i - 1] * (fieldsH + lastH) / 2.0
        
        lastE = fieldsE
        lastH = fieldsH
    return resE, resH

def WavesIntegrator2D(wave1, wave2, fieldsFunc):
    resE, resH = None, None
    lastE, lastH = None, None
    dPhis = wave1.phis[1:] - wave1.phis[:-1] 

    for i in range(len(wave1.betas)):
        fieldsFunc2 = lambda w, nr: fieldsFunc(wave1, i, w, nr)
        fieldsE, fieldsH = WavesIntegrator1D(wave2, fieldsFunc2)
        
        if len(wave1.betas) == 1:
            return fieldsE, fieldsH
        
        if i == 0:
            resE = np.zeros_like(fieldsE)
            resH = np.zeros_like(fieldsH)    
        else:
            resE += dPhis[i - 1] * (fieldsE + lastE) / 2.0
            resH += dPhis[i - 1] * (fieldsH + lastH) / 2.0
        
        lastE = fieldsE
        lastH = fieldsH
        
    return resE, resH

#===============================================================================
# TmmForWavePackets
#===============================================================================

class TmmForWaves(Core.ParamsBaseClass):

    def __init__(self, tmm, wave, **kwargs):
        self._params = ["th0", "pol"]
        self.wave = wave 
        self.tmm = tmm
        self.th0 = None
        self.pol = None
        super(TmmForWaves, self).__init__(**kwargs)
    
    def CalcFields2d(self, xs, ys, **kwargs):
        
        def IntegFunc(wave, nr):
            self.tmm.SetParam(beta = wave.betas[nr])
            E, H = self.tmm.CalcFields2D(xs, ys, self.pol)
            E *= wave.expansionCoefsPhi[nr]
            H *= wave.expansionCoefsPhi[nr]
            return E, H
        
        self.SetParams(**kwargs)
        wl = self.tmm.GetParam("wl").real
        nPrism = self.tmm._materialsCache[0][1]
        self.wave.Solve(wl, self.th0, n = nPrism)

        ESum, HSum = WavesIntegrator1D(self.wave, IntegFunc)
        return ESum, HSum

#===============================================================================
# SecondOrderNLTmmForWaves
#===============================================================================

class SecondOrderNLTmmForWaves(Core.ParamsBaseClass):
    
    def __init__(self, tmmNL, wave1, wave2, **kwargs):
        self._params = ["thP1", "thP2"]
        self.tmmNL = tmmNL
        self.wave1 = wave1
        self.wave2 = wave2
        self.thP1 = None
        self.thP2 = None
        
        super(SecondOrderNLTmmForWaves, self).__init__(**kwargs)
    
    # Powers (porks only in linear media!)

    def CalcPowersP1(self, layerNr, x0, x1, z, **kwargs):
        self.SetParams(**kwargs)
        res = self._CalcPowersPump(self.tmmNL.tmmP1, self.thP1, self.wave1, layerNr, x0, x1, z)
        return res

    def CalcPowersP2(self, x0, x1, z, **kwargs):
        self.SetParams(**kwargs)
        res = self._CalcPowersPump(self.tmmNL.tmmP2, self.thP2, self.wave2, x0, x1, z)
        return res
    
    def CalcPowersGen(self, layerNr, x0, x1, z, **kwargs):
        self.SetParams(**kwargs)
        
        # Solve wave
        starttime = time()
        self.wave1.Solve(self.tmmNL.wlP1, self.thP1, n = self.tmmNL.layers[0][1])
        self.wave2.Solve(self.tmmNL.wlP2, self.thP2, n = self.tmmNL.layers[0][1])
        print("wave.Solve", time() - starttime)
        
        # Init variables
        starttime = time()
        kxs = np.zeros((len(self.wave1.kxs) * len(self.wave2.kxs)))
        UsF = np.zeros_like(kxs, dtype = complex)
        UsB = np.zeros_like(kxs, dtype = complex)
        kzsF = np.zeros_like(kxs, dtype = complex)
        kzsB = np.zeros_like(kxs, dtype = complex)
        print("init variables", time() - starttime)
        
        # Solve system for every beta
        starttime = time()
        for nr1 in range(len(self.wave1.kxs)):
            for nr2 in range(len(self.wave2.kxs)): 
                self.tmmNL.SetParams(betaP1 = self.wave1.betas[nr1],
                                     betaP2 = self.wave2.betas[nr2],
                                     overrideE0P1 = self.wave1.expansionCoefsKx[nr1],
                                     overrideE0P2 = self.wave2.expansionCoefsKx[nr2])
                self.tmmNL.Solve()
                
                nr = nr1 * len(self.wave2.kxs) + nr2
                kxs[nr] = self.tmmNL.betaGen * 2.0 * np.pi / self.tmmNL.wlGen
                U = self.tmmNL.tmmGen.layers[layerNr].hw.GetMainFields(0.0)
                UsF[nr], UsB[nr] = U[0, 0], U[1, 0]
                kzsF[nr] = self.tmmNL.tmmGen.layers[layerNr].hw.kzF
                kzsB[nr] = self.tmmNL.tmmGen.layers[layerNr].hw.kzB
        print("solve system", time() - starttime)
                
        # Sort by kx
        starttime = time()
        sortP = kxs.argsort()
        kxs = kxs[sortP]
        UsF = UsF[sortP]
        UsB = UsB[sortP]
        kzsF = kzsF[sortP]
        kzsB = kzsB[sortP]
        print("sorting", time() - starttime)
        
        Ly = self.wave1.Ly
        if self.wave2.Ly != Ly:
            raise ValueError("All input waves must have same Ly")
        
        PF = self._IntegrateWavePower(self.tmmNL.tmmGen, UsF, kxs, kzsF, layerNr, x0, x1, z, Ly)
        PB = -self._IntegrateWavePower(self.tmmNL.tmmGen, UsB, kxs, kzsB, layerNr, x0, x1, z, Ly)
        return PF, PB
        
        
    # Fields
    
    def CalcFields2dP1(self, zs, xs, **kwargs):
        self.SetParams(**kwargs)
        res = self._CalcFields2dPump(self.tmmNL.tmmP1, self.thP1, self.wave1, zs, xs)
        return res

    def CalcFields2dP2(self, zs, xs, **kwargs):
        self.SetParams(**kwargs)
        res = self._CalcFields2dPump(self.tmmNL.tmmP2, self.thP2, self.wave2, zs, xs)
        return res        
    
    def CalcFields2dGen(self, zs, xs, **kwargs):
        def IntegFunc(wave1, nr1, wave2, nr2):
            self.tmmNL.SetParams(betaP1 = wave1.betas[nr1],
                                 betaP2 = wave2.betas[nr2],
                                 overrideE0P1 = wave1.expansionCoefsPhi[nr1],
                                 overrideE0P2 = wave2.expansionCoefsPhi[nr2])
            self.tmmNL.Solve()
            E, H = self.tmmNL.tmmGen.GetFields2D(zs, xs)
            return E, H

        self.SetParams(**kwargs)
        self.wave1.Solve(self.tmmNL.wlP1, self.thP1, n = self.tmmNL.layers[0][1])
        self.wave2.Solve(self.tmmNL.wlP2, self.thP2, n = self.tmmNL.layers[0][1])
        ESum, HSum = WavesIntegrator2D(self.wave1, self.wave2, IntegFunc)
        return ESum, HSum

    # Private methods

    def _CalcFields2dPump(self, tmmP, thP, wave, zs, xs):
        def IntegFunc(wave, nr):
            tmmP.SetParams(beta = wave.betas[nr], overrideE0 = wave.expansionCoefsPhi[nr])
            tmmP.Solve()
            E, H = tmmP.GetFields2D(zs, xs)
            return E, H
        
        wave.Solve(tmmP.wl, thP, n = tmmP.layers[0].n)
        ESum, HSum = WavesIntegrator1D(wave, IntegFunc)
        return ESum, HSum
    
    def _CalcPowersPump(self, tmmP, thP, wave, layerNr, x0, x1, z):
        # Solve wave
        starttime = time()
        wave.Solve(tmmP.wl, thP, n = tmmP.layers[0].n)
        print("wave.Solve", time() - starttime)
        
        # Init variables
        starttime = time()
        kxs = wave.kxs
        UsF = np.zeros_like(kxs, dtype = complex)
        UsB = np.zeros_like(kxs, dtype = complex)
        kzsF = np.zeros_like(kxs, dtype = complex)
        kzsB = np.zeros_like(kxs, dtype = complex)
        print("init variables", time() - starttime)
        
        # Solve system for every beta
        starttime = time()
        for nr in range(len(kxs)): 
            tmmP.SetParams(beta = wave.betas[nr], overrideE0 = wave.expansionCoefsKx[nr])
            tmmP.Solve()
            U = tmmP.layers[layerNr].hw.GetMainFields(0.0)
            UsF[nr], UsB[nr] = U[0, 0], U[1, 0]
            kzsF[nr] = tmmP.layers[layerNr].hw.kzF
            kzsB[nr] = tmmP.layers[layerNr].hw.kzB
        print("solve system", time() - starttime)
        
        PF = self._IntegrateWavePower(tmmP, UsF, kxs, kzsF, layerNr, x0, x1, z, wave.Ly)
        PB = -self._IntegrateWavePower(tmmP, UsB, kxs, kzsB, layerNr, x0, x1, z, wave.Ly)
        return PF, PB
            
    def _Integrate(self, xs, values):
        dxs = xs[1:] - xs[:-1]
        res = np.sum(dxs * 0.5 * (values[1:] + values[:-1]))
        return res
    
    def _IntegrateWavePower(self, tmm, Us, kxs, kzs, layerNr, x0, x1, z, Ly):
        # Integrate wave
        starttime = time()    
        
        if len(kxs) == 1:
            # Plane wave
            integ2d = Us[0] * np.conj(Us[0]) * (x1 - x0) * kzs[0]
        else:
            kxP = kxs
            kzP = kzs
            
            # Internal integral
            integ1ds = np.zeros_like(kxs, dtype = complex)
            for i, (kx, kz) in enumerate(zip(kxs, kzs)):
                # Fx
                dk = kx - kxP
                dk[i] = 1.0
                Fx = -1.0j / dk * (np.exp(1.0j * dk * x1) - np.exp(1.0j * dk * x0))
                Fx[i] = x1 - x0
                dk[i] = 0.0
                
                kzPart = kz if tmm.pol == "p" else kzP
                integValues = np.conj(Us) * Fx * kzPart * np.exp(1.0j * (kz - kzP) * z)     
                integ1ds[i] = self._Integrate(kxs, integValues)
 
            # Outer integral
            integ2d = self._Integrate(kxs,  Us * integ1ds)
            
        omega = Constants.WlToOmega(tmm.wl)
        if tmm.pol == "p":
            P = Ly / (2.0 * omega * Constants.eps0) * (1.0 / tmm.layers[layerNr].eps * integ2d).real
        elif tmm.pol == "s":
            P = Ly / (2.0 * omega * Constants.mu0) * (integ2d).real
        else:
            raise NotImplementedError()
        print("integration", time() - starttime)
        
        return P
    
if __name__ == "__main__":
    pass