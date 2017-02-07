import numpy as np
from CppSecondOrderNLTMM cimport *

#===============================================================================
# Common
#===============================================================================

cdef PolarizationCpp PolarizationFromStr(str polStr):
    if polStr == "p":
        return P_POL
    elif polStr == "s":
        return S_POL
    else:
        raise NotImplementedError()
    
cdef WaveDirectionCpp WaveDirectionFromStr(str waveStr):
    if waveStr == "forward":
        return F
    elif waveStr == "backward":
        return B
    elif waveStr == "total":
        return TOT
    else:
        raise NotImplementedError()

cdef NonlinearTmmModeCpp NonlinearTmmModeFromStr(str modeStr):
    if modeStr == "incident":
        return MODE_INCIDENT
    elif modeStr == "nonlinear":
        return MODE_NONLINEAR
    else:
        raise ValueError()
    
cdef NonlinearProcessCpp NonlinearProcessFromStr(str processStr):
    if processStr == "sfg":
        return SFG
    elif processStr == "dfg":
        return DFG
    else:
        raise ValueError()

cdef paramDict = {"wl": PARAM_WL, \
                  "beta": PARAM_BETA, \
                  "pol": PARAM_POL, \
                  "I0": PARAM_I0, \
                  "overrideE0": PARAM_OVERRIDE_E0, \
                  "E0": PARAM_E0, \
                  "mode": PARAM_MODE}
    
cdef TMMParamCpp TmmParamFromStr(str paramStr):
    return paramDict[paramStr]
    
#===============================================================================
# Chi2Tensor
#===============================================================================

cdef class _Chi2Tensor:
    cdef Chi2TensorCpp* _thisptr;
    cdef object _parent;
    
    def __cinit__(self):
        # This object is always allocated by Material class
        _thisptr = NULL
    
    def __dealloc__(self):
        # This object is always deallocated by Material class
        pass
    
    cdef _Init(self, Chi2TensorCpp* ptr, object parent):
        self._thisptr = ptr
        self._parent = parent # Avoid dealloc of parent
    
    def Update(self, **kwargs):
        cdef int i1, i2, i3;
        cdef double value;
        
        # d-values
        for k in list(kwargs.keys()):
            if (not k.startswith("d")) or (len(k) != 3):
                continue
            
            i1 = int(k[1])
            i2 = int(k[2])
            value = kwargs.pop(k)
            self._thisptr.SetD(i1, i2, value)
            
        # chi2-values
        for k in list(kwargs.keys()):
            if (not k.startswith("chi")) or (len(k) != 6):
                continue
            i1 = int(k[1])
            i2 = int(k[2])
            i3 = int(k[3])
            value = kwargs.pop(k)
            self._thisptr.SetChi2(i1, i2, i3, value)
        
        # Distinct fields
        cdef bool distinctFields
        if "distinctFields" in kwargs:
            distinctFields = kwargs.pop("distinctFields")
            self._thisptr.SetDistinctFields(distinctFields)
        
        # Phis (may generate exception if no phiY-Z available)
        cdef double phiX, phiY, phiZ
        if "phiX" in kwargs:
            phiX = kwargs.pop("phiX")
            phiY = kwargs.pop("phiY")
            phiZ = kwargs.pop("phiZ")
            self._thisptr.SetRotation(phiX, phiY, phiZ)
        
        # Errors
        for k in list(kwargs.keys()):
            raise ValueError("Unknown kwarg %s" % (k))
        
    def GetChi2Tensor(self):
        cdef int i, j, k
        cdef np.ndarray res = np.zeros([3, 3, 3], dtype = float)
        
        # Eigency currently doesn't support Tensors
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    res[i, j, k] = self._thisptr.GetChi2Element(i + 1, j + 1, k + 1)
        return res
    
#===============================================================================
# Material
#===============================================================================

cdef class Material:
    cdef MaterialCpp *_thisptr
    cdef readonly object chi2

    def __cinit__(self, np.ndarray[double, ndim = 1] wls, np.ndarray[double complex, ndim = 1] ns):
        # Copy is made in c++
        self._thisptr = new MaterialCpp(Map[ArrayXd](wls), Map[ArrayXcd](ns))
        
        cdef Chi2TensorCpp *chi2Ptr = &self._thisptr.chi2
        chi2 = _Chi2Tensor()
        chi2._Init(chi2Ptr, self) # This class is passed as parent (avoids dealloc before chi2)
        self.chi2 = chi2
                
    def __dealloc__(self):
        del self._thisptr
        
    def __call__(self, wl): # For compatibility
        return self.GetN(wl)
        
    def GetN(self, wl):
        return self._thisptr.GetN(wl)
        
    def IsNonlinear(self):
        return self._thisptr.IsNonlinear()

#===============================================================================
# PowerFlows
#===============================================================================

cdef class _PowerFlows:
    cdef readonly double complex inc, r, t
    cdef readonly double I, R, T
    
    def __cinit__(self):
        pass
    
    cdef _Init(self, PowerFlowsCpp *ptr):
        self.inc = ptr.inc
        self.r = ptr.r
        self.t = ptr.t
        self.I = ptr.I
        self.R = ptr.R
        self.T = ptr.T

#===============================================================================
# SweepResultNonlinearTMM
#===============================================================================

cdef class _SweepResultNonlinearTMM:
    cdef SweepResultNonlinearTMMCpp *_thisptr 
    cdef readonly np.ndarray inc, r, t
    cdef readonly np.ndarray I, R, T
    cdef bool _needDealloc;
    cdef object _parent;
    
    def __cinit__(self):
        self._needDealloc = False
        pass
    
    def __dealloc__(self):
        if self._needDealloc:
            del self._thisptr

    cdef _Init(self, SweepResultNonlinearTMMCpp *ptr, bool needDealloc = True, object parent = None):
        self._thisptr = ptr
        self._needDealloc = needDealloc
        self._parent = parent # To avoid dealloc of parent before this
        
        self.inc = ndarray_view(self._thisptr.inc).squeeze()
        self.r = ndarray_view(self._thisptr.r).squeeze()
        self.t = ndarray_view(self._thisptr.t).squeeze()
        self.I = ndarray_view(self._thisptr.I).squeeze()
        self.R = ndarray_view(self._thisptr.R).squeeze()
        self.T = ndarray_view(self._thisptr.T).squeeze()

   
#===============================================================================
# FieldsZ
#===============================================================================
    
cdef class _FieldsZ:
    cdef FieldsZCpp *_thisptr; 
    cdef readonly np.ndarray E, H;

    def __cinit__(self):
        pass
    
    def __dealloc__(self):
        del self._thisptr

    cdef _Init(self, FieldsZCpp *ptr):
        self._thisptr = ptr

        self.E = ndarray_view(self._thisptr.E)
        self.H = ndarray_view(self._thisptr.H)

#===============================================================================
# FieldsZX
#===============================================================================
    
cdef class _FieldsZX:
    cdef FieldsZXCpp *_thisptr; 
    cdef readonly np.ndarray Ex, Ey, Ez, Hx, Hy, Hz;

    def __cinit__(self):
        pass
    
    def __dealloc__(self):
        del self._thisptr

    cdef _Init(self, FieldsZXCpp *ptr):
        self._thisptr = ptr
        cdef PolarizationCpp pol = self._thisptr.GetPol()
        
        if pol == P_POL:
            self.Hy = ndarray_view(self._thisptr.Hy)
            self.Ex = ndarray_view(self._thisptr.Ex)
            self.Ez = ndarray_view(self._thisptr.Ez)
            self.Hx = None
            self.Ey = None
            self.Hz = None
        elif pol == S_POL:
            self.Hx = ndarray_view(self._thisptr.Hx)
            self.Ey = ndarray_view(self._thisptr.Ey)
            self.Hz = ndarray_view(self._thisptr.Hz)
            self.Hy = None
            self.Ex = None
            self.Ez = None
        else:
            raise ValueError("Unknown polarization.")

#===============================================================================
# HomogeneousWave
#===============================================================================

cdef class _HomogeneousWave:
    cdef HomogeneousWaveCpp *_thisptr
    
    def __cinit__(self):
        self._thisptr = NULL
    
    cdef _Init(self, HomogeneousWaveCpp *ptr):
        self._thisptr = ptr
        
    def GetMainFields(self, double z):
        # Have to copy, because result is in stack
        return ndarray_copy(self._thisptr.GetMainFields(z))
        
    @property
    def kzF(self):
        return self._thisptr.GetKzF()

    @property
    def kx(self):
        return self._thisptr.GetKx()

#===============================================================================
# NonlinearLayer
#===============================================================================

cdef class _NonlinearLayer:
    cdef NonlinearLayerCpp *_thisptr
    cdef readonly object hw
    cdef object _parent;
    cdef int _layerNr;
    
    def __cinit__(self):
        self._thisptr = NULL
        self.hw = None
    
    cdef _Init(self, NonlinearLayerCpp *ptr, int layerNr, object parent):
        self._thisptr = ptr
        self._parent = parent
        self._layerNr = layerNr 
        
        hw = _HomogeneousWave()
        hw._Init(self._thisptr.GetHw())
        self.hw = hw
        
    def GetPowerFlow(self, double z):
        return self._thisptr.GetPowerFlow(z);
    
    def GetAbsorbedPower(self):
        return self._thisptr.GetAbsorbedPower();
    
    def GetSrcPower(self):
        return self._thisptr.GetSrcPower();
     
    def GetPowerFlowsForWave(self, object wave, double th0, double x0, double x1, double z, str dirStr = "total"):
        return self._parent._GetPowerFlowsForWave(wave, th0, self._layerNr, x0, x1, z, dirStr)
     
#===============================================================================
# NonlinearTMM
#===============================================================================

cdef class NonlinearTMM:
    cdef NonlinearTMMCpp *_thisptr
    cdef list materialsCache
    cdef readonly list layers
    cdef bool _needDealloc
    cdef object _parent;
    
    def __cinit__(self, bool initStruct = True, object parent = None, **kwargs):
        if initStruct:
            self._thisptr = new NonlinearTMMCpp()
            self._needDealloc = True
        else:
            self._thisptr = NULL
            self._needDealloc = False
        self._parent = parent
        
        self.materialsCache = []
        self.layers = []
        self.SetParams(**kwargs)
    
    cdef _Init(self, NonlinearTMMCpp* ptr):
        self._thisptr = ptr
    
    def __dealloc__(self):
        if self._needDealloc:
            del self._thisptr

    # Methods
    #---------------------------------------------------------------------------
        
    def AddLayer(self, double d, Material material):
        # No copy of material is made
        self._thisptr.AddLayer(d, material._thisptr)
        
        # Cache material classes, avoids dealloc
        self.materialsCache.append(material)
        
        # Layers list
        cdef int lastLayer = self._thisptr.LayersCount() - 1
        cdef NonlinearLayerCpp *layerptr = self._thisptr.GetLayer(lastLayer)
        layer = _NonlinearLayer()
        layer._Init(layerptr, lastLayer, self)
        self.layers.append(layer)
        
    def SetParams(self, **kwargs):
        for name, value in kwargs.iteritems():
            if name not in paramDict:
                raise ValueError("Unknown kwarg %s" % (name))
            setattr(self, name, value)
        
    def Solve(self, **kwargs):
        self.SetParams(**kwargs)
        self._thisptr.Solve()
            
    def GetPowerFlows(self):
        cdef PowerFlowsCpp resCpp = self._thisptr.GetPowerFlows()
        res = _PowerFlows()
        res._Init(&resCpp)
        return res 
    
    def Sweep(self, str paramStr, np.ndarray[double, ndim = 1] values):
        cdef SweepResultNonlinearTMMCpp *resCpp;
        resCpp = self._thisptr.Sweep(TmmParamFromStr(paramStr), Map[ArrayXd](values))
        res = _SweepResultNonlinearTMM()
        res._Init(resCpp);
        return res
    
    def GetFields(self, np.ndarray[double, ndim = 1] zs, str dir = "total"):
        cdef FieldsZCpp *resCpp;
        resCpp = self._thisptr.GetFields(Map[ArrayXd](zs), WaveDirectionFromStr(dir))
        res = _FieldsZ()
        res._Init(resCpp)
        return res
    
    def GetFields2D(self, np.ndarray[double, ndim = 1] zs, np.ndarray[double, ndim = 1] xs, str dir = "total"):
        cdef FieldsZXCpp *resCpp;
        resCpp = self._thisptr.GetFields2D(Map[ArrayXd](zs), Map[ArrayXd](xs), WaveDirectionFromStr(dir))
        res = _FieldsZX()
        res._Init(resCpp)
        return res
    
    def GetWaveFields2D(self, np.ndarray[double, ndim = 1] betas, \
            np.ndarray[double complex, ndim = 1] E0s, np.ndarray[double, ndim = 1] zs, \
            np.ndarray[double, ndim = 1] xs, str dirStr = "total"):
        
        cdef FieldsZXCpp *resCpp;
        cdef WaveDirectionCpp direction = WaveDirectionFromStr(dirStr)
        resCpp = self._thisptr.GetWaveFields2D(Map[ArrayXd](betas), Map[ArrayXcd](E0s), \
            Map[ArrayXd](zs), Map[ArrayXd](xs), direction)
        res = _FieldsZX()
        res._Init(resCpp)
        return res
    
    def GetAbsorbedPower(self):
        return self._thisptr.GetAbsorbedPower();
    
    def _GetPowerFlowsForWave(self, object wave, double th0, int layerNr, double x0, double x1, double z, str dirStr = "total"):
        # NonlinearLayer has its own specific method
        cdef double Ly = wave.Ly
        cdef WaveDirectionCpp direction = WaveDirectionFromStr(dirStr)
        wave.Solve(self.wl, th0)
        
        cdef pair[double, double] res;
        res = self._thisptr.GetPowerFlowsForWave(Map[ArrayXd](wave.betas), \
            Map[ArrayXcd](wave.expansionCoefsKx), layerNr, x0, x1, z, Ly, direction)
        
        return (res.first, res.second)
    
    # Getters
    #--------------------------------------------------------------------------- 
    
    @property
    def wl(self):
        return self._thisptr.GetDouble(PARAM_WL)    
    
    @property
    def beta(self):
        return self._thisptr.GetDouble(PARAM_BETA)    

    @property
    def pol(self):
        return "ps"[self._thisptr.GetInt(PARAM_POL)]    

    @property
    def I0(self):
        return self._thisptr.GetDouble(PARAM_I0)    

    @property
    def overrideE0(self):
        return self._thisptr.GetBool(PARAM_OVERRIDE_E0)    

    @property
    def E0(self):
        return self._thisptr.GetComplex(PARAM_E0)    

    @property
    def mode(self):
        return ["incident", "nonlinear"][self._thisptr.GetComplex(PARAM_MODE)]    
            
    # Setter
    #--------------------------------------------------------------------------- 
        
    @wl.setter
    def wl(self, value):  # @DuplicatedSignature
        self._thisptr.SetParam(PARAM_WL, <double>value)
        
    @beta.setter
    def beta(self, value):  # @DuplicatedSignature
        self._thisptr.SetParam(PARAM_BETA, <double>value)
    
    @pol.setter
    def pol(self, polStr):  # @DuplicatedSignature
        self._thisptr.SetParam(PARAM_POL, <int>PolarizationFromStr(polStr))
        
    @I0.setter
    def I0(self, value):  # @DuplicatedSignature
        self._thisptr.SetParam(PARAM_I0, <double>value)
            
    @overrideE0.setter
    def overrideE0(self, value):  # @DuplicatedSignature
        self._thisptr.SetParam(PARAM_OVERRIDE_E0, <bool>value)
        
    @E0.setter
    def E0(self, value):  # @DuplicatedSignature
        self._thisptr.SetParam(PARAM_E0, <double complex>value)
        
    @mode.setter
    def mode(self, modeStr):  # @DuplicatedSignature
        self._thisptr.SetParam(PARAM_MODE, <int>NonlinearTmmModeFromStr(modeStr))

#===============================================================================
# SecondOrderNLPowerFlows
#===============================================================================

cdef class _SecondOrderNLPowerFlows:
    cdef readonly object P1, P2, Gen
    
    def __cinit__(self):
        pass
    
    cdef _Init(self, SecondOrderNLPowerFlowsCpp* ptr):
        P1 = _PowerFlows()
        P1._Init(&ptr.P1)
    
        P2 = _PowerFlows()
        P2._Init(&ptr.P2)
        
        Gen = _PowerFlows()
        Gen._Init(&ptr.Gen)
        
        self.P1 = P1
        self.P2 = P2
        self.Gen = Gen
    
#===============================================================================
# SweepResultSecondOrderNLTMM
#===============================================================================
    
cdef class _SweepResultSecondOrderNLTMM:
    cdef SweepResultSecondOrderNLTMMCpp *_thisptr
    cdef readonly object P1, P2, Gen
    
    def __cinit__(self):
        self._thisptr = NULL
        pass
    
    def __dealloc__(self):
        if self._thisptr:
            del self._thisptr   

    cdef _Init(self, SweepResultSecondOrderNLTMMCpp *ptr):
        self._thisptr = ptr
        
        P1 = _SweepResultNonlinearTMM()
        P2 = _SweepResultNonlinearTMM()
        Gen = _SweepResultNonlinearTMM()
        
        P1._Init(&ptr.P1, False, self)
        P2._Init(&ptr.P2, False, self)
        Gen._Init(&ptr.Gen, False, self)
        
        self.P1 = P1
        self.P2 = P2
        self.Gen = Gen
        
#===============================================================================
# SecondOrderNLTMM
#===============================================================================

cdef class SecondOrderNLTMM:
    cdef SecondOrderNLTMMCpp *_thisptr
    cdef list materialsCache
    cdef readonly object P1, P2, Gen

    def __cinit__(self, str mode = "sfg"):
        self._thisptr = new SecondOrderNLTMMCpp()
        self._thisptr.SetProcess(NonlinearProcessFromStr(mode))
        self.materialsCache = []
        
        # Pointers to TMM-s
        cdef NonlinearTMMCpp* tmmP1Ptr = self._thisptr.GetP1()
        cdef NonlinearTMMCpp* tmmP2Ptr = self._thisptr.GetP2()
        cdef NonlinearTMMCpp* tmmGenPtr = self._thisptr.GetGen()
        
        # Wrapper classes
        cdef NonlinearTMM tmmP1Py = NonlinearTMM(False)
        cdef NonlinearTMM tmmP2Py = NonlinearTMM(False)
        cdef NonlinearTMM tmmGenPy = NonlinearTMM(False)
    
        # Init wrappers
        tmmP1Py._Init(tmmP1Ptr)
        tmmP2Py._Init(tmmP2Ptr)
        tmmGenPy._Init(tmmGenPtr)
        
        # Save to members
        self.P1 = tmmP1Py
        self.P2 = tmmP2Py
        self.Gen = tmmGenPy 
    
    def __dealloc__(self):
        del self._thisptr
    
    # Methods
    #---------------------------------------------------------------------------
        
    def AddLayer(self, double d, Material material):
        # No copy of material is made
        self._thisptr.AddLayer(d, material._thisptr)
        
        # Cache material classes, avoids material dealloc
        self.materialsCache.append(material)
        
    def Solve(self):
        self._thisptr.Solve()
        
    def GetPowerFlows(self):
        cdef SecondOrderNLPowerFlowsCpp resCpp;
        resCpp = self._thisptr.GetPowerFlows();
        
        res = _SecondOrderNLPowerFlows()
        res._Init(&resCpp)
        return res
        
    def Sweep(self, str paramStr, np.ndarray[double, ndim = 1] valuesP1, np.ndarray[double, ndim = 1] valuesP2):
        cdef SweepResultSecondOrderNLTMMCpp *resCpp;
        resCpp = self._thisptr.Sweep(TmmParamFromStr(paramStr), Map[ArrayXd](valuesP1), Map[ArrayXd](valuesP2))
        res = _SweepResultSecondOrderNLTMM()
        res._Init(resCpp);
        return res
    
    def GetGenWaveFields2D(self, np.ndarray[double, ndim = 1] betasP1, np.ndarray[double, ndim = 1] betasP2,
                        np.ndarray[double complex, ndim = 1] E0sP1, np.ndarray[double complex, ndim = 1] E0sP2,
                        np.ndarray[double, ndim = 1] zs, np.ndarray[double, ndim = 1] xs, str dirStr = "total"):
        
        cdef FieldsZXCpp *resCpp;
        cdef WaveDirectionCpp direction = WaveDirectionFromStr(dirStr)
        resCpp = self._thisptr.GetGenWaveFields2D(Map[ArrayXd](betasP1), Map[ArrayXd](betasP2),
            Map[ArrayXcd](E0sP1), Map[ArrayXcd](E0sP2), Map[ArrayXd](zs), Map[ArrayXd](xs), direction)
        res = _FieldsZX()
        res._Init(resCpp)
        return res
        