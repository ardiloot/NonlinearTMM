import numpy as np

__all__ = ["PlaneWave",
           "GaussianWave",
           "TukeyWaveFFT"]


const_c = 299792458.0
eps0 = 8.854187817e-12
mu0 = 1.2566370614e-6

#===============================================================================
# PlaneWave
#===============================================================================

class PlaneWave(object):
    
    def __init__(self, params = [], **kwargs):
        paramsThis = ["pwr", "overrideE0", "w0", "n", "Ly"]
        self.overrideE0 = None #E0 specified in vacuum
        self.Ly = None
        self._params = paramsThis + params
        self.SetParams(**kwargs)

    def SetParams(self, **kwargs):
        for k in kwargs:
            if k not in self._params:
                raise ValueError("Unknown params: %s" % (k))
        
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        

    def Solve(self, wl, th0, **kwargs):
        self.wl = wl
        self.th0 = th0
        self.SetParams(**kwargs)
        
        self._Precalc()
        self._Solve()

    def _Precalc(self): 
        if self.overrideE0 is None:
            self.I0 = self.pwr / (self.Ly * self.w0) 
            self.E0 = np.sqrt(2.0 * mu0 * const_c * self.I0) 
        else:
            self.E0 = self.overrideE0
            
        self.k0 = 2.0 * np.pi / self.wl
        self.k = self.k0 * self.n(self.wl).real
        
    def _Solve(self):
        self.phis = np.array([0.0])
        self.kxs = np.array([self.k * np.sin(self.th0)])
        self.kzs = np.array([self.k * np.cos(self.th0)])
        self.expansionCoefsPhi = np.array([self.E0], dtype = complex)
        self.expansionCoefsKx = np.array([self.E0], dtype = complex)
        self.betas = (self.kxs / self.k0).real
   
#===============================================================================
# GaussianWave
#===============================================================================

class GaussianWave(PlaneWave):
    def __init__(self, params = [], **kwargs):
        paramsThis = ["integCriteria", "nPointsInteg"]
        self.nPointsInteg = 30
        self.integCriteria = 1e-3
        super().__init__(params + paramsThis, **kwargs)
    
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
        self.expansionCoefsKx = profileSpectrum / np.cos(self.th0)
        self.betas = (self.kxs / self.k0).real

#===============================================================================
# _WaveFFT
#===============================================================================

class _WaveFFT(PlaneWave):
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

class GaussianWaveFFT(_WaveFFT):
    def __init__(self, params = [], **kwargs):
        paramsThis = []
        _WaveFFT.__init__(self, params + paramsThis, **kwargs)
    
    def _FieldProfile(self, xs):
        res = self.E0 * np.exp(- xs ** 2.0 / self.w0 ** 2.0)
        return res

#===============================================================================
# TukeyWaveFFT
#===============================================================================

class TukeyWaveFFT(_WaveFFT):
    def __init__(self, params = [], **kwargs):
        paramsThis = ["a"]
        self.a = 0.5
        _WaveFFT.__init__(self, params + paramsThis, **kwargs)
    
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
        
if __name__ == "__main__":
    pass