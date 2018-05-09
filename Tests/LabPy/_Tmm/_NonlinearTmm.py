import numpy as np
from LabPy import Constants, Core
from LabPy.Constants import WlToOmega, OmegaToWl

__all__ = ["Chi2Tensor", "SecondOrderNLTmm"]

X = 0
Y = 1
Z = 2

#===============================================================================
# Chi2Tensor
#===============================================================================

class Chi2Tensor(object):
    
    def __init__(self, dtype = complex, distinctFields = True, **kwargs):
        self._dIndices = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        self._chi2 = np.zeros((3, 3, 3), dtype = dtype)
        self._chi2Rotated = None
        self._chi2RotatedChanged = True
        
        self.distinctFields = distinctFields
        self.phiX = 0.0
        self.phiY = 0.0
        self.phiZ = 0.0
        
        self.Update(**kwargs)
        
        
    def Update(self, **kwargs):
        self._chi2RotatedChanged = True
        self.phiX = float(kwargs.pop("phiX", self.phiX))
        self.phiY = float(kwargs.pop("phiY", self.phiY))
        self.phiZ = float(kwargs.pop("phiZ", self.phiZ))
        
        # d-values
        for k in list(kwargs.keys()):
            if (not k.startswith("d")) or (len(k) != 3):
                continue
            i1, (i2, i3) = int(k[1]) - 1, self._dIndices[int(k[2]) - 1]
            self._chi2[i1, i2, i3] = 2.0 * kwargs.pop(k)
            self._chi2[i1, i3, i2] = self._chi2[i1, i2, i3]
            
        # chi2-values
        for k in list(kwargs.keys()):
            if (not k.startswith("chi")) or (len(k) != 6):
                continue
            i1, i2, i3 = int(k[3]) - 1, int(k[4]) - 1, int(k[5]) - 1
            self._chi2[i1, i2, i3] = kwargs.pop(k)
        
        # Errors
        for k in list(kwargs.keys()):
            raise ValueError("Unknown kwarg %s" % (k))
        

    def GetChi2(self):
        if not self._chi2RotatedChanged:
            return self._chi2Rotated
        
        # Compute chi2Rotated
        self._chi2Rotated = Chi2Tensor._RotateTensorO3(self._chi2, self.phiX, self.phiY, self.phiZ)
        self._chi2RotatedChanged = False
        return self._chi2Rotated
    
    def GetD(self):
        res = np.zeros((3, 6), dtype = self._chi2.dtype)
        chi2 = self.GetChi2()
        
        for i in range(3):
            for j in range(6):
                a, b = self._dIndices[j]
                if abs(chi2[i, a, b] - chi2[i, b, a]) > 1e-20:
                    raise ValueError("Not equal elements %s, %s" % ((i, a, b), (i, b, a)))
                
                res[i, j] = chi2[i, a, b]
        return res
        
    def GetNonlinearPolarization(self, E1, E2):
        chi2 = self.GetChi2()
        fieldTensor = np.outer(E1, E2)
        res = Constants.eps0 * np.tensordot(chi2, fieldTensor).flatten()
        
        if self.distinctFields:
            res *= 2.0
        return res
        
    def GetChi2Eff(self, E1, E2, E3):
        E1Norm = (E1 / np.linalg.norm(E1)).flatten()
        E2Norm = (E2 / np.linalg.norm(E2)).flatten()
        E3Norm = (E3 / np.linalg.norm(E3)).flatten()
        p = self.GetNonlinearPolarization(E1Norm, E2Norm)

        res = np.dot(E3Norm, p / Constants.eps0)
        if self.distinctFields:
            res *= 2.0
        return res
        
    def GetDEff(self, E1, E2, E3):
        chi2Eff = self.GetChi2Eff(E1, E2, E3)
        res = chi2Eff / 2.0
        return res
        
    @staticmethod        
    def _RotateTensorO3(tensor, phiX = 0.0, phiY = 0.0, phiZ = 0.0):
    
        def ApplyRotationMatrix(inp, R):
            res = np.zeros_like(inp)
            for i1 in range(3):
                for i2 in range(3):
                    for i3 in range(3):
                        for j1 in range(3):
                            for j2 in range(3):
                                for j3 in range(3):
                                    res[i1, i2, i3] += R[i1, j1] * R[i2, j2] * R[i3, j3] * inp[j1, j2, j3]
                                    
            return res
        
        Rx = Core.RotationMatrixX(phiX)
        Ry = Core.RotationMatrixY(phiY)
        Rz = Core.RotationMatrixZ(phiZ)
        
        res = ApplyRotationMatrix(tensor, Rx)
        res = ApplyRotationMatrix(res, Ry)
        res = ApplyRotationMatrix(res, Rz)
            
        return res

#===============================================================================
# _HomogeneousWave
#===============================================================================

class _HomogeneousWave(object):
    
    def __init__(self, layer):
        self.layer = layer
        
    def Solve(self, kx):
        layer = self.layer
        
        self.kx = kx
        self.k = self.layer.k0 * self.layer.n(self.layer.wl) 
        self.kzF = np.sqrt(self.k ** 2.0 - self.kx ** 2.0, dtype = complex)
        self.kzB = -self.kzF
        self.kz = np.array([[self.kzF, self.kzB]]).T
        
        if self.kzF.imag < 0.0:
            raise ValueError("kzF imaginary part negative %s" % (self.kzF))
        
        # Propagation matrix

        if layer.d == float("inf"):
            self.propMatrix = np.identity(2, dtype = complex)
        else:
            phase = np.exp(1.0j * self.kzF * layer.d)
            self.propMatrix = np.array([[phase, 0.0],
                                        [0.0, 1.0 / phase]],
                                        dtype = complex)
            
    def GetMainFields(self, zs):
        U0 = np.array([[self.layer.fieldAmps[0, 0], self.layer.fieldAmps[1, 0]]]).T
        U = U0 * np.exp(1.0j * self.kz * zs)
        return U
    

#===============================================================================
# _InhomogeneosWave
#===============================================================================

class _InhomogeneosWave(object):
    
    def __init__(self, layer):
        self.layer = layer
        
    def Solve(self, kx, kpS):
        layer = self.layer
        kSzF, pF, pB = [0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] if kpS == None else kpS 
        
        self.kx = kx
        self.kSzF = kSzF
        self.kSzB = -kSzF
        self.kSz = np.array([[self.kSzF, self.kSzB]]).T
        self.kS = np.sqrt(self.kx ** 2.0 + self.kSzF ** 2.0, dtype = complex)
        
        self.pxF, self.pyF, self.pzF = [0.0, 0.0, 0.0] if pF is None else pF
        self.pxB, self.pyB, self.pzB = [0.0, 0.0, 0.0] if pB is None else pB
        self.px = np.array([[self.pxF, self.pxB]]).T
        self.py = np.array([[self.pyF, self.pyB]]).T
        self.pz = np.array([[self.pzF, self.pzB]]).T
        
        if self.kS.imag < 0.0:
            #print ValueError("kS imaginary part negative %s" % (self.kS))
            self.kS *= -1.0
        
        # Check if layer is nonlinear
        
        if layer.pol == "s":
            self.isNonlinearF = abs(self.pyF) > 1e-200
            self.isNonlinearB = abs(self.pyB) > 1e-200
        elif layer.pol == "p":
            self.isNonlinearF = abs(self.pxF) > 1e-200 or abs(self.pzF) > 1e-200
            self.isNonlinearB = abs(self.pxB) > 1e-200 or abs(self.pzB) > 1e-200
        else:
            raise NotImplementedError("Unknown polarization.")
        self.isNonlinear = self.isNonlinearF or self.isNonlinearB
        
        # Calc B-s
        
        if not self.isNonlinear:
            self.ByF = 0.0
            self.ByB = 0.0
        else:
            if layer.pol == "s":
                BFunc = lambda py: -(py / Constants.eps0) * layer.k0 ** 2.0 / \
                    (layer.k ** 2.0 - self.kS ** 2.0)
                self.ByF = BFunc(self.pyF)
                self.ByB = BFunc(self.pyB) * np.exp(1.0j * self.kSzF * layer.d)
            elif layer.pol == "p":
                BFunc = lambda px, pz, kSz: -layer.omega / (layer.k ** 2.0 - self.kS ** 2.0) * \
                    (kSz * px - self.kx * pz) 
                self.ByF = BFunc(self.pxF, self.pzF, self.kSzF)
                self.ByB = BFunc(self.pxB, self.pzB, -self.kSzF) * np.exp(1.0j * self.kSzF * layer.d)
            else:
                raise NotImplementedError("Unknown polarization.")
        self.By = np.array([[self.ByF, self.ByB]]).T
        
        # Propagation Matrix NL
        
        if layer.d == float("inf"):
            self.propMatrixNL = np.array([[0.0, 0.0]], dtype = complex).T
            if self.isNonlinear:
                print(NotImplementedError("First and last medium must be linear.!"))
        else:
            if self.isNonlinear:
                self.propMatrixNL = self.By * (np.exp(1.0j * self.kSz * layer.d) - \
                                               np.exp(1.0j * layer.hw.kz * layer.d))
            else:
                self.propMatrixNL = np.array([[0.0, 0.0]], dtype = complex).T       
                
    def GetMainFields(self, zs):
        U = self.By * (np.exp(1.0j * self.kSz * zs) - np.exp(1.0j * self.layer.hw.kz * zs))
        return U
             

#===============================================================================
# _NonlinearLayer
#===============================================================================
    
class _NonlinearLayer(Core.ParamsBaseClass):

    def __init__(self, d, n, **kwargs):
        self._params = ["d", "n", "kpS", "kpA"]
        self.d = d
        self.n = n
        self.kpS = None
        self.kpA = None
    
        self.hw = _HomogeneousWave(self)
        self.iws = _InhomogeneosWave(self)
        self.iwa = _InhomogeneosWave(self)
         
        
        super(_NonlinearLayer, self).__init__(**kwargs)
                        
    def Solve(self, wl, beta, pol, lastLayer = False):
        self.wl = wl
        self.beta = beta
        self.pol = pol
        self.omega = Constants.WlToOmega(wl)
        self.lastLayer = lastLayer
        
        # Find wavevectors

        self.eps = self.n(wl) ** 2.0
        self.k0 = 2.0 * np.pi / wl
        self.k = self.k0 * self.n(self.wl)
        kx = beta * self.k0

        # Solve waves

        self.hw.Solve(kx)
        self.iws.Solve(kx, self.kpS)
        self.iwa.Solve(kx, self.kpA)
        
        # Propagation matrix
        
        self.propMatrix = self.hw.propMatrix
        self.propMatrixNL = self.iws.propMatrixNL + self.iwa.propMatrixNL
        
    def GetMainFields(self, zs):
        return self.hw.GetMainFields(zs) + self.iws.GetMainFields(zs) + self.iwa.GetMainFields(zs)
    
    def GetFields(self, zs, outBackAndForward = False):
        E = np.zeros((2, len(zs), 3), dtype = complex)
        H = np.zeros((2, len(zs), 3), dtype = complex)
        
        expKSz = np.exp(1.0j * self.iws.kSz * zs)
        expKAz = np.exp(1.0j * self.iwa.kSz * zs)
        if self.pol == "p":
            c = -1.0 / (self.omega * self.eps * Constants.eps0)
            Hy = self.GetMainFields(zs)
            
            H[:, :, Y] = Hy
            E[:, :, X] = c * (-self.hw.kz * Hy + \
                self.iws.By * (self.hw.kz - self.iws.kSz) * expKSz + self.omega * self.iws.px * expKSz + \
                self.iwa.By * (self.hw.kz - self.iwa.kSz) * expKAz + self.omega * self.iwa.px * expKAz)
            E[:, :, Z] =  c * (self.hw.kx * Hy + \
                self.omega * self.iws.pz * expKSz + \
                self.omega * self.iwa.pz * expKAz)
        elif self.pol == "s":
            c = 1.0 / (self.omega * Constants.mu0)
            Ey = self.GetMainFields(zs)
            
            E[:, :, Y] = Ey
            H[:, :, X] = c * (-self.hw.kz * Ey + \
                self.iws.By * (self.hw.kz - self.iws.kSz) * expKSz + \
                self.iwa.By * (self.hw.kz - self.iwa.kSz) * expKAz)
            H[:, :, Z] = c * (self.hw.kx * Ey)
        else:
            raise NotImplementedError("Unknown polarization.")
        
        if outBackAndForward:
            return E, H
        else:
            return (E[0, :, :] + E[1, :, :]), (H[0, :, :] + H[1, :, :])
        
    def GetAmplitudes(self, z = 0.0):
        (E0FTemp, E0BTemp), (H0FTemp, H0BTemp) = self.GetFields(np.array([z]), outBackAndForward = True)
        
        E0F, E0B = E0FTemp[0, :], E0BTemp[0, :]
        H0F, H0B = H0FTemp[0, :], H0BTemp[0, :]
        return (E0F, E0B), (H0F, H0B)
    
    def GetPowerFlow(self, z):
        (E0F, E0B), (H0F, H0B) = self.GetAmplitudes(z = z)
        E, H = E0F + E0B, H0F + H0B
        Sz = 0.5 * (E[0] * np.conj(H[1]) - E[1] * np.conj(H[0])).real
        return Sz
    
    def GetAbsorbedPower(self):
        if self.eps.imag == 0.0:
            # If layer is not absorbing
            return 0.0
        
        (E0F, E0B), (_, __) = self.GetAmplitudes()
        kz = self.hw.kz[0, 0]
        intValue1 = -1.0j * (np.exp(1.0j * (kz - np.conj(kz)) * self.d) - 1.0) / (kz - np.conj(kz))
        intValue2 = -1.0j * (np.exp(1.0j * (kz + np.conj(kz)) * self.d) - 1.0) / (kz + np.conj(kz))
        intValue3 = 1.0j * (np.exp(-1.0j * (kz + np.conj(kz)) * self.d) - 1.0) / (kz + np.conj(kz))
        intValue4 = 1.0j * (np.exp(-1.0j * (kz - np.conj(kz)) * self.d) - 1.0) / (kz - np.conj(kz))
        
        intValue = np.dot(E0F, np.conj(E0F)) * intValue1 + \
                   np.dot(E0F, np.conj(E0B)) * intValue2 + \
                   np.dot(E0B, np.conj(E0F)) * intValue3 + \
                   np.dot(E0B, np.conj(E0B)) * intValue4
        
        res = 0.5 * Constants.eps0 * self.eps.imag * self.omega * intValue
        return res.real
    
    def GetSrcPower(self):
        absorbed = self.GetAbsorbedPower()
        if self.d == float("inf"):
            deltaS = 0.0
        else:
            deltaS = self.GetPowerFlow(self.d) - self.GetPowerFlow(0.0)
            
        res = absorbed + deltaS
        return res
        
#===============================================================================
# _NonlinearTmm
#===============================================================================
                                         
class _NonlinearTmm(Core.ParamsBaseClass):
    MODE_INCIDENT = "incident"
    MODE_NONLINEAR = "nonlinear"
    
    def __init__(self, **kwargs):
        self._params = ["wl", "beta", "pol", "I0", "overrideE0", "mode"]
        self.I0 = 1e10
        self.overrideE0 = None #E0 specified in vacuum, not in first layer
        self.layers = []
        
        super(_NonlinearTmm, self).__init__(**kwargs)
    
    def AddLayer(self, *args, **kwargs):
        self.layers.append(_NonlinearLayer(*args, **kwargs))
    
    def Solve(self, **kwargs):
        self.SetParams(**kwargs)
        
        # Solve all layers
        
        for i in range(len(self.layers)):
            last = (i == len(self.layers) - 1)
            self.layers[i].Solve(self.wl, self.beta, self.pol, last)
        # Solve system matrix
        
        mSysL, mSysNL = self.__SystemMatrix(len(self.layers))
        
        # Calc reflection and transmission
        
        l0 = self.layers[0]
        lL = self.layers[-1]
        omega = Constants.WlToOmega(self.wl)
        if self.mode == self.MODE_INCIDENT:
            if self.overrideE0 is None:
                cosTh0 = l0.hw.kz[0, 0].real / l0.k.real
                if self.pol == "p":
                    inc = np.sqrt(abs(2.0 * omega * Constants.eps0 / l0.hw.kzF.real * l0.eps.real * self.I0 * cosTh0), dtype = complex)
                elif self.pol == "s":
                    inc = np.sqrt(abs(2.0 * omega * Constants.mu0 / l0.hw.kzF.real * self.I0 *  cosTh0), dtype = complex)
                else:
                    raise NotImplementedError("Unknown polarization.")
            else:
                if self.pol == "p":
                    inc = self.overrideE0 * np.sqrt(l0.n(self.wl).real) * Constants.eps0 * Constants.c
                elif self.pol == "s":
                    inc = self.overrideE0 / np.sqrt(l0.n(self.wl).real)
                else:
                    raise NotImplementedError("Unknown polarization.")
                
            r = -inc * mSysL[1, 0] / mSysL[1, 1]
            t = inc * np.linalg.det(mSysL) / mSysL[1, 1]
        elif self.mode == self.MODE_NONLINEAR:
            inc = 0.0
            r = -mSysNL[1, 0] / mSysL[1, 1]
            t = mSysNL[0, 0] - mSysL[0, 1] / mSysL[1, 1] * mSysNL[1, 0]
        else:
            raise NotImplementedError("Unknown mode.")
            
        # Calc power flows
        
        if self.pol == "p":
            c = 1.0 / (2.0 * omega * Constants.eps0)
            I = c * l0.hw.kzF.real / l0.eps.real * abs(inc) ** 2.0
            R = c * l0.hw.kzF.real / l0.eps.real * abs(r) ** 2.0
            T = c * lL.hw.kzF.real / lL.eps.real * abs(t) ** 2.0
        elif self.pol == "s":
            c = 1.0 / (2.0 * omega * Constants.mu0)
            I = c * l0.hw.kzF.real * abs(inc) ** 2.0
            R = c * l0.hw.kzF.real * abs(r) ** 2.0
            T = c * lL.hw.kzF.real * abs(t) ** 2.0
        else:
            raise NotImplementedError("Unknown polarization.")
        
        # Calc field coefs in all layers
        
        self.layers[0].fieldAmps = np.array([[inc, r]], dtype = complex).T
        for i in range(1, len(self.layers)):
            mL, mNL = self.__SystemMatrix(i + 1)
            self.layers[i].fieldAmps = np.dot(mL, self.layers[0].fieldAmps) + mNL

        # Absorbed energy
        A = self.GetAbsorbedPower()
               
        return inc, r, t, I, R, T, A
        
            
    def __TransferMatrix(self, layerNr):
        
        def TransferMatrixNL(w1, w2):
            exp1F = np.exp(1.0j * w1.kSzF * l1.d) if l1.d != float("inf") else 0.0
            exp1B = np.exp(-1.0j * w1.kSzF * l1.d) if l1.d != float("inf") else 0.0
            #exp1B = 1.0 / exp1F if l1.d != float("inf") else 0.0
            
            f1 = (w1.kSzF - l1.hw.kzF) * (w1.ByF * exp1F - w1.ByB * exp1B)
            f2 = (w2.kSzF - l2.hw.kzF) * (w2.ByB - w2.ByF)
            
            if self.pol == "p":
                f3 = -l1.omega * l2.eps * (w1.pxB * exp1B + w1.pxF * exp1F) + \
                    l1.omega * l1.eps * (w2.pxB + w2.pxF)
                cNL = 1.0 / (2.0 * l1.eps * l2.hw.kzF) * (l2.eps * f1 + l1.eps * f2 + f3)
            elif self.pol == "s":
                cNL = 1.0 / (2.0 * l2.hw.kzF) * (f1 + f2)
            else:
                raise NotImplementedError("Unknown polarization.")
            
            transferMatrixNL = np.array([[cNL, -cNL]], dtype = complex).T
            return transferMatrixNL
        
        l1, l2 = self.layers[layerNr], self.layers[layerNr + 1]
        
        # Polarization specific
        
        if self.pol == "p":
            a = (l2.eps * l1.hw.kzF) / (l1.eps * l2.hw.kzF)
        elif self.pol == "s":
            a = l1.hw.kzF / l2.hw.kzF
        else:
            raise NotImplementedError("Unknown polarization.")
        
        # Transfer matrices
        transferMatrix = 0.5 * np.array([[1.0 + a, 1.0 - a], [1.0 - a, 1.0 + a]], dtype = complex)
        transferMatrixNL = TransferMatrixNL(l1.iws, l2.iws) + TransferMatrixNL(l1.iwa, l2.iwa)
        
        return transferMatrix, transferMatrixNL

    def __SystemMatrix(self, nLayers):
        # Independent of polarization
        
        mSysL = np.identity(2, dtype = complex)
        mSysNL = np.array([[0.0, 0.0]], dtype = complex).T
        
        for i in range(nLayers - 1):
            layer = self.layers[i]
            transferL, transferNL = self.__TransferMatrix(i)
            
            mSysL = np.dot(layer.propMatrix, mSysL) 
            mSysL = np.dot(transferL, mSysL)
            
            mSysNL = np.dot(layer.propMatrix, mSysNL) + layer.propMatrixNL
            mSysNL = np.dot(transferL, mSysNL) + transferNL

        return mSysL, mSysNL
    
    def GetFields(self, zs, outBackAndForward = False): 
        if not outBackAndForward:
            resE = np.zeros((len(zs), 3), dtype = complex)
            resH = np.zeros((len(zs), 3), dtype = complex)
        else:
            resE = np.zeros((2, len(zs), 3), dtype = complex)
            resH = np.zeros((2, len(zs), 3), dtype = complex)
        layerDsIndexes = self.__GetLayerIndexesAndDistances(zs)

        for layerId, layerDist, bIndex, eIndex in layerDsIndexes:
            if bIndex == eIndex:
                continue
            
            E, H = self.layers[layerId].GetFields(zs[bIndex:eIndex] - layerDist, outBackAndForward = outBackAndForward)
            if not outBackAndForward:
                resE[bIndex:eIndex, :] = E
                resH[bIndex:eIndex, :] = H
            else:
                resE[:, bIndex:eIndex, :] = E
                resH[:, bIndex:eIndex, :] = H
        
        return resE, resH
    
    def GetFields2D(self, zs, xs, outBackAndForward = False):
        E1D, H1D = self.GetFields(zs, outBackAndForward = outBackAndForward)
        phaseX = np.exp(1.0j * 2.0 * np.pi / self.wl * self.beta * xs)
        
        
        if not outBackAndForward:
            resE = np.zeros((len(zs), len(xs), 3), dtype = complex)
            resH = np.zeros((len(zs), len(xs), 3), dtype = complex)
            
            for i in range(3):
                resE[:, :, i] = np.outer(E1D[:, i], phaseX)
                resH[:, :, i] = np.outer(H1D[:, i], phaseX)
            
        
        else:
            resE = np.zeros((2, len(zs), len(xs), 3), dtype = complex)
            resH = np.zeros((2, len(zs), len(xs), 3), dtype = complex)
            
            for j in range(2):
                for i in range(3):
                    resE[j, :, :, i] = np.outer(E1D[j, :, i], phaseX)
                    resH[j, :, :, i] = np.outer(H1D[j, :, i], phaseX)
                    
        return resE, resH
    
    def GetAbsorbedPower(self):
        res = 0.0
        for layer in self.layers:
            res += layer.GetAbsorbedPower()
        return res
    
    def __GetLayerIndexesAndDistances(self, zs):
        layerDsIndexes = [] 
        curLayer, curDist, layerDist, beginIndex = 0, 0.0, 0.0, 0
        for i in range(len(zs) + 1):
            if (i == len(zs)) or (zs[i] >= curDist):
                layerDsIndexes.append((curLayer, layerDist, beginIndex, i))
                if i == len(zs):
                    break
                layerDist += self.layers[curLayer].d if curLayer > 0 else 0.0
                curLayer += 1
                curDist += self.layers[curLayer].d
                beginIndex = i
        return layerDsIndexes

#===============================================================================
# SecondOrderNLTmm
#===============================================================================

class SecondOrderNLTmm(Core.ParamsBaseClass):
    SFG = "sfg"
    DFG = "dfg"
    
    def __init__(self, **kwargs):
        self._params = ["I0P1", "wlP1", "betaP1", "polP1", "overrideE0P1", \
                        "I0P2", "wlP2", "betaP2", "polP2", "overrideE0P2", \
                        "polGen", "process"]
        self.layers = []
        self.I0P1 = 1e10
        self.I0P2 = 1e10
        self.overrideE0P1 = None
        self.overrideE0P2 = None
        self.process = self.SFG
        super(self.__class__, self).__init__(**kwargs)
    
    def AddLayer(self, d, n, chi2 = Chi2Tensor()):
        self.layers.append((d, n, chi2))
    
    def Solve(self, **kwargs):
        self.SetParams(**kwargs)
        
        self._SolveFundamentalFields()
        
        if self.process == self.SFG:
            self.wlGen = OmegaToWl(WlToOmega(self.wlP1) + WlToOmega(self.wlP2))
            self.betaGen =  self.wlGen * (self.betaP1 / self.wlP1 + self.betaP2 / self.wlP2)
            kSzFunc = lambda kzFP1, kzFP2: kzFP1 + kzFP2
            kAzFunc = lambda kzFP1, kzFP2: kzFP1 - kzFP2
            pFunc = lambda chi2, E1, E2: chi2.GetNonlinearPolarization(E1, E2)
        elif self.process == self.DFG:
            self.wlGen = OmegaToWl(WlToOmega(self.wlP1) - WlToOmega(self.wlP2))
            self.betaGen =  self.wlGen * (self.betaP1 / self.wlP1 - self.betaP2 / self.wlP2)
            kSzFunc = lambda kzFP1, kzFP2: kzFP1 - np.conj(kzFP2)
            kAzFunc = lambda kzFP1, kzFP2: kzFP1 + np.conj(kzFP2)
            pFunc = lambda chi2, E1, E2: chi2.GetNonlinearPolarization(E1, np.conj(E2))
        else:
            raise ValueError("Unknown process: %s" % (self.process))
        
        self.tmmGen = _NonlinearTmm(wl = self.wlGen, \
                               pol = self.polGen, \
                               mode = _NonlinearTmm.MODE_NONLINEAR)
        
        for i in range(len(self.layers)):
            d, n, chi2 = self.layers[i]
            (E0P1F, E0P1B), _ = self.tmmP1.layers[i].GetAmplitudes()
            (E0P2F, E0P2B), _ = self.tmmP2.layers[i].GetAmplitudes()
            kzFP1, kzFP2 = self.tmmP1.layers[i].hw.kzF, self.tmmP2.layers[i].hw.kzF
            
            kSz, kAz = kSzFunc(kzFP1, kzFP2), kAzFunc(kzFP1, kzFP2) 
            pSF, pSB = pFunc(chi2, E0P1F, E0P2F), pFunc(chi2, E0P1B, E0P2B)
            pAF, pAB = pFunc(chi2, E0P1F, E0P2B), pFunc(chi2, E0P1B, E0P2F)
            
            if kSz.imag < 0.0:
                #print "reversed kSz"
                kSz, pSF, pSB = -kSz, pSB, pSF 

            if kAz.imag < 0.0:
                #print "reversed kAz"
                kAz, pAF, pAB = -kAz, pAB, pAF
            
            self.tmmGen.AddLayer(d, n, kpS = (kSz, pSF, pSB), kpA = (kAz, pAF, pAB))
            
        self.resGen = self.tmmGen.Solve(beta = self.betaGen)
        
        return self.resP1, self.resP2, self.resGen
    
    def Sweep(self, paramNames, paramValues, enhpos = None):
        keysP1 = ["iP1", "rP1", "tP1", "IP1", "RP1", "TP1", "AP1"]
        keysP2 = ["iP2", "rP2", "tP2", "IP2", "RP2", "TP2", "AP2"]
        keysGen = ["iGen", "rGen", "tGen", "IGen", "RGen", "TGen", "AGen"]
        res = {}
        
        res["betasGen"] = np.zeros((len(paramValues[0])))
        
        for k in keysP1 + keysP2 + keysGen:
            res[k] = np.zeros((len(paramValues[0])), dtype = complex)
        
        if enhpos is not None:
            res["enhP1"] = np.zeros((len(paramValues[0])))
            res["enhP2"] = np.zeros((len(paramValues[0])))
        
        for i in np.arange(len(paramValues[0])):
            for j in range(len(paramNames)):
                self.SetParams(**{paramNames[j]: paramValues[j][i]})
            
            rrP1, rrP2, rrGen = self.Solve()
            res["betasGen"][i] = self.tmmGen.beta.real
            
            for j in np.arange(len(keysP1)):
                res[keysP1[j]][i] = rrP1[j]
            for j in np.arange(len(keysP2)):
                res[keysP2[j]][i] = rrP2[j]
            for j in np.arange(len(keysGen)):
                res[keysGen[j]][i] = rrGen[j]
            
            if enhpos is not None:
                enhLayer, enhDist = enhpos 
                EP1, _ = self.tmmP1.layers[enhLayer].GetFields(np.array([enhDist]))
                (E0P1, _), __ = self.tmmP1.layers[0].GetFields(np.array([0.0]), outBackAndForward = True)
                enhP1 = np.linalg.norm(EP1[0, :]) / (np.linalg.norm(E0P1[0, :]) * np.sqrt(self.tmmP1.layers[0].n(self.wlP1).real))
                
                EP2, _ = self.tmmP2.layers[enhLayer].GetFields(np.array([enhDist]))
                (E0P2, _), __ = self.tmmP2.layers[0].GetFields(np.array([0.0]), outBackAndForward = True)
                enhP2 = np.linalg.norm(EP2[0, :]) / (np.linalg.norm(E0P2[0, :]) * np.sqrt(self.tmmP2.layers[0].n(self.wlP2).real))
                
                res["enhP1"][i] = enhP1
                res["enhP2"][i] = enhP2

        return res
    
    def _SolveFundamentalFields(self):
        # Solve fundametal field 1
        self.tmmP1 = _NonlinearTmm(I0 = self.I0P1,
                             overrideE0 = self.overrideE0P1,
                             wl = self.wlP1, \
                             pol = self.polP1, \
                             mode = _NonlinearTmm.MODE_INCIDENT)
        
        for d, n, _ in self.layers:
            self.tmmP1.AddLayer(d, n)
        self.resP1 = self.tmmP1.Solve(beta = self.betaP1)

        # Solve fundametal field 2
        self.tmmP2 = _NonlinearTmm(I0 = self.I0P2,
                             overrideE0 = self.overrideE0P2,
                             wl = self.wlP2, \
                             pol = self.polP2, \
                             mode = _NonlinearTmm.MODE_INCIDENT)
        
        for d, n, _ in self.layers:
            self.tmmP2.AddLayer(d, n)
        self.resP2 = self.tmmP2.Solve(beta = self.betaP2)

if __name__ == "__main__":
    pass
