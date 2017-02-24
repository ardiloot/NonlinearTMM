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

cdef WaveTypeCpp WaveTypeFromStr(str waveTypeStr):
    if waveTypeStr == "planewave":
        return PLANEWAVE
    if waveTypeStr == "gaussian":
        return GAUSSIANWAVE
    if waveTypeStr == "tukey":
        return TUKEYWAVE
    else:
        raise NotImplementedError()

cdef paramDict = {"wl": PARAM_WL, \
                  "beta": PARAM_BETA, \
                  "pol": PARAM_POL, \
                  "I0": PARAM_I0, \
                  "overrideE0": PARAM_OVERRIDE_E0, \
                  "E0": PARAM_E0, \
                  "mode": PARAM_MODE,
                  "w0": PARAM_WAVE_W0}
    
cdef waveParamsSet = set(["waveType",
                "pwr",
                "overrideE0",
                "E0",
                "w0",
                "Ly",
                "a",
                "nPointsInteg",
                "maxX",
                "dynamicMaxX",
                "dynamicMaxXCoef",
                "maxPhi"])

    
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
# Waves
#===============================================================================

cdef class _Wave:
    cdef WaveCpp *_thisptr
    cdef bool _needsDealloc;
    
    def __cinit__(self):
        self._thisptr = NULL
        self._needsDealloc = False;
        
    cdef _Init(self, WaveCpp *ptr):
        self._thisptr = ptr
        self._needsDealloc = False
        
    def __dealloc__(self):
        if self._needsDealloc:
            del self._thisptr
        
    def SetParams(self, **kwargs):
        for name, value in kwargs.iteritems():
            if name not in waveParamsSet:
                raise ValueError("Unknown kwarg %s" % (name))
            setattr(self, name, value)
        
    # Getters
    #--------------------------------------------------------------------------- 
        
    @property
    def waveType(self):
        raise NotImplementedError()
        
    @property
    def pwr(self):
        return self._thisptr.GetPwr()
    
    @property
    def overrideE0(self):
        return self._thisptr.GetOverrideE0()
    
    @property
    def E0(self):
        return self._thisptr.GetE0()

    @property
    def w0(self):
        return self._thisptr.GetW0()
    
    @property
    def Ly(self):
        return self._thisptr.GetLy()
    
    @property
    def a(self):
        return self._thisptr.GetA()
    
    @property
    def nPointsInteg(self):
        return self._thisptr.GetNPointsInteg()

    @property
    def maxX(self):
        return self._thisptr.GetMaxX()
        
    @property
    def dynamicMaxX(self):
        return self._thisptr.IsDynamicMaxXEnabled()
    
    @property
    def dynamicMaxXCoef(self):
        return self._thisptr.GetDynamicMaxXCoef()

    @property
    def maxPhi(self):
        return self._thisptr.GetMaxPhi()

    @property
    def xRange(self):
        cdef pair[double, double] r = self._thisptr.GetXRange()
        return (r.first, r.second) 

    @property
    def betas(self):
        return ndarray_copy(self._thisptr.GetBetas()).squeeze()

    @property
    def phis(self):
        return ndarray_copy(self._thisptr.GetPhis()).squeeze()
    
    @property
    def kxs(self):
        return ndarray_copy(self._thisptr.GetKxs()).squeeze()

    @property
    def kzs(self):
        return ndarray_copy(self._thisptr.GetKzs()).squeeze()
    
    @property
    def fieldProfile(self):
        xs = ndarray_copy(self._thisptr.GetFieldProfileXs()).squeeze()
        fieldProfile = ndarray_copy(self._thisptr.GetFieldProfile()).squeeze()
        return xs, fieldProfile
    
    @property
    def expansionCoefsKx(self):
        return ndarray_copy(self._thisptr.GetExpansionCoefsKx()).squeeze()

    # Setters
    #---------------------------------------------------------------------------

    @waveType.setter
    def waveType(self, str waveTypeStr):  # @DuplicatedSignature
        self._thisptr.SetWaveType(WaveTypeFromStr(waveTypeStr))
    
    @pwr.setter    
    def pwr(self, double value):  # @DuplicatedSignature
        self._thisptr.SetPwr(value)

    @overrideE0.setter
    def overrideE0(self, bool value): # @DuplicatedSignature
        self._thisptr.SetOverrideE0(value)

    @E0.setter
    def E0(self, double value): # @DuplicatedSignature
        self._thisptr.SetE0(value)
        
    @w0.setter
    def w0(self, double value): # @DuplicatedSignature
        self._thisptr.SetW0(value)
        
    @Ly.setter
    def Ly(self, double value): # @DuplicatedSignature
        self._thisptr.SetLy(value)
    
    @a.setter
    def a(self, double value): # @DuplicatedSignature
        self._thisptr.SetA(value)
        
    @nPointsInteg.setter
    def nPointsInteg(self, int value): # @DuplicatedSignature
        self._thisptr.SetNPointsInteg(value)

    @maxX.setter
    def maxX(self, double value): # @DuplicatedSignature
        self._thisptr.SetMaxX(value)  
    
    @dynamicMaxX.setter
    def dynamicMaxX(self, bool value): # @DuplicatedSignature
        self._thisptr.EnableDynamicMaxX(value)  
            
    @dynamicMaxXCoef.setter
    def dynamicMaxXCoef(self, double value): # @DuplicatedSignature
        self._thisptr.SetDynamicMaxXCoef(value)    
    
    @maxPhi.setter
    def maxPhi(self, double value): # @DuplicatedSignature
        self._thisptr.SetMaxPhi(value)    
     

#===============================================================================
# Intensities
#===============================================================================

cdef class _Intensities:
    cdef readonly double complex inc, r, t
    cdef readonly double I, R, T
    
    def __cinit__(self):
        pass
    
    cdef _Init(self, IntensitiesCpp *ptr):
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
    cdef readonly np.ndarray Ii, Ir, It, Ia, enh
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
        self.Ii = ndarray_view(self._thisptr.Ii).squeeze()
        self.Ir = ndarray_view(self._thisptr.Ir).squeeze()
        self.It = ndarray_view(self._thisptr.It).squeeze()
        self.Ia = ndarray_view(self._thisptr.Ia).squeeze()
        self.enh = ndarray_view(self._thisptr.enh).squeeze()


#===============================================================================
# SweepResultNonlinearTMM
#===============================================================================

cdef class _WaveSweepResultNonlinearTMM:
    cdef WaveSweepResultNonlinearTMMCpp *_thisptr 
    cdef readonly np.ndarray Pi, Pr, Pt, enh, beamArea
    cdef bool _needDealloc;
    cdef object _parent;
    
    def __cinit__(self):
        self._needDealloc = False
        pass
    
    def __dealloc__(self):
        if self._needDealloc:
            del self._thisptr

    cdef _Init(self, WaveSweepResultNonlinearTMMCpp *ptr, bool needDealloc = True, object parent = None):
        self._thisptr = ptr
        self._needDealloc = needDealloc
        self._parent = parent # To avoid dealloc of parent before this
   
        self.Pi = ndarray_view(self._thisptr.Pi).squeeze()
        self.Pr = ndarray_view(self._thisptr.Pr).squeeze()
        self.Pt = ndarray_view(self._thisptr.Pt).squeeze()
        self.enh = ndarray_view(self._thisptr.enh).squeeze()
        self.beamArea = ndarray_view(self._thisptr.beamArea).squeeze()
        

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
        
    @property
    def EN(self):
        cdef PolarizationCpp pol = self._thisptr.GetPol()
        if pol == P_POL:
            return np.sqrt(self.Ex.real ** 2 + self.Ex.imag ** 2 + \
                           self.Ez.real ** 2 + self.Ez.imag ** 2)
        elif pol == S_POL:
            return np.sqrt(self.Ey.real ** 2 + self.Ey.imag ** 2)
        else:
            raise ValueError("Unknown polarization.")

    @property
    def HN(self):
        cdef PolarizationCpp pol = self._thisptr.GetPol()
        if pol == P_POL:
            return np.sqrt(self.Hy.real ** 2 + self.Hy.imag ** 2)
        elif pol == S_POL:
            return np.sqrt(self.Hx.real ** 2 + self.Hx.imag ** 2 + \
                           self.Hz.real ** 2 + self.Hz.imag ** 2)
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
        
    def GetIntensity(self, double z):
        return self._thisptr.GetIntensity(z);
    
    def GetAbsorbedIntensity(self):
        return self._thisptr.GetAbsorbedIntensity();
    
    def GetSrcIntensity(self):
        return self._thisptr.GetSrcIntensity();
    
    # Getter
    #--------------------------------------------------------------------------- 
    
    @property
    def d(self):
        return self._thisptr.GetThickness()    
            
    # Setter
    #--------------------------------------------------------------------------- 
        
    @d.setter
    def d(self, value):  # @DuplicatedSignature
        self._thisptr.SetThickness(value)
    
    
#===============================================================================
# NonlinearTMM
#===============================================================================

cdef class NonlinearTMM:
    cdef NonlinearTMMCpp *_thisptr
    cdef readonly list materialsCache
    cdef readonly list layers
    cdef bool _needDealloc
    cdef object _parent;
    cdef readonly _Wave wave;
    
    def __cinit__(self, bool initStruct = True, object parent = None, **kwargs):
        if initStruct:
            self._thisptr = new NonlinearTMMCpp()
            self._Init(self._thisptr)
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
        cdef WaveCpp *wavePtr = self._thisptr.GetWave() 
        wave = _Wave()
        wave._Init(wavePtr)
        self.wave = wave
    
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
            
    def GetIntensities(self):
        cdef IntensitiesCpp resCpp = self._thisptr.GetIntensities()
        res = _Intensities()
        res._Init(&resCpp)
        return res 
    
    def Sweep(self, str paramStr, np.ndarray[double, ndim = 1] values, int layerNr = 0, double layerZ = 0.0, bool outPwr = True, bool outAbs = False, outEnh = False):
        cdef SweepResultNonlinearTMMCpp *resCpp;
        cdef int outmask = 0
        
        if outPwr:
            outmask |= SWEEP_PWRFLOWS
        if outAbs:
            outmask |= SWEEP_ABS
        if outEnh:
            outmask |= SWEEP_ENH
            
        cdef TMMParamCpp param;
        cdef int paramLayer = -1;
        if (paramStr.startswith("d_")):
            param = PARAM_LAYER_D
            paramLayer = int(paramStr[2:])
        else:
            param = TmmParamFromStr(paramStr)
        
        resCpp = self._thisptr.Sweep(TmmParamFromStr(paramStr), Map[ArrayXd](values), outmask, paramLayer, layerNr, layerZ)
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
    
    def GetAbsorbedIntensity(self):
        return self._thisptr.GetAbsorbedIntensity();
    
    def GetEnhancement(self, int layerNr, double z = 0.0):
        cdef double res
        res = self._thisptr.GetEnhancement(layerNr, z)
        return res
    
    # Waves
    
    def WaveGetPowerFlows(self, int layerNr, double x0 = float("nan"), double x1 = float("nan"), double z = 0.0):
        # NonlinearLayer has its own specific method
        cdef pair[double, double] res;
        res = self._thisptr.WaveGetPowerFlows(layerNr, x0, x1, z)
        return (res.first, res.second)
    
    def WaveGetEnhancement(self, int layerNr, double z = 0.0):
        cdef double res
        res = self._thisptr.WaveGetEnhancement(layerNr, z)
        return res
    
    def WaveSweep(self, str paramStr, np.ndarray[double, ndim = 1] values, \
            int layerNr = 0, double layerZ = 0.0, bool outPwr = True, \
            outR = False, outT = False, outEnh = False):
        
        cdef WaveSweepResultNonlinearTMMCpp *resCpp;
        cdef int outmask = 0
        
        if outPwr:
            outmask |= SWEEP_PWRFLOWS
        if outEnh:
            outmask |= SWEEP_ENH
        if outR:
            outmask |= SWEEP_R
        if outT:
            outmask |= SWEEP_T
            
        cdef TMMParamCpp param;
        cdef int paramLayer = -1;
        if (paramStr.startswith("d_")):
            param = PARAM_LAYER_D
            paramLayer = int(paramStr[2:])
        else:
            param = TmmParamFromStr(paramStr)
            
        resCpp = self._thisptr.WaveSweep(TmmParamFromStr(paramStr), Map[ArrayXd](values), outmask, paramLayer, layerNr, layerZ)
        res = _WaveSweepResultNonlinearTMM()
        res._Init(resCpp);
        return res
           
    def WaveGetFields2D(self, np.ndarray[double, ndim = 1] zs, \
            np.ndarray[double, ndim = 1] xs, str dirStr = "total"):
        
        cdef FieldsZXCpp *resCpp;
        cdef WaveDirectionCpp direction = WaveDirectionFromStr(dirStr)
        resCpp = self._thisptr.WaveGetFields2D(Map[ArrayXd](zs), Map[ArrayXd](xs), direction)
        res = _FieldsZX()
        res._Init(resCpp)
        return res
    
    # Getters
    #--------------------------------------------------------------------------- 
    
    @property
    def wl(self):
        return self._thisptr.GetWl()  
    
    @property
    def beta(self):
        return self._thisptr.GetBeta()   

    @property
    def pol(self):
        return "ps"[<int>self._thisptr.GetPolarization()]    

    @property
    def I0(self):
        return self._thisptr.GetI0()

    @property
    def overrideE0(self):
        return self._thisptr.GetOverrideE0()   

    @property
    def E0(self):
        return self._thisptr.GetE0()    

    @property
    def mode(self):
        return ["incident", "nonlinear"][<int>self._thisptr.GetMode()]    
            
    # Setter
    #--------------------------------------------------------------------------- 
        
    @wl.setter
    def wl(self, value):  # @DuplicatedSignature
        self._thisptr.SetWl(<double>value)
        
    @beta.setter
    def beta(self, value):  # @DuplicatedSignature
        self._thisptr.SetBeta(<double>value)
    
    @pol.setter
    def pol(self, polStr):  # @DuplicatedSignature
        self._thisptr.SetPolarization(PolarizationFromStr(polStr))
        
    @I0.setter
    def I0(self, value):  # @DuplicatedSignature
        self._thisptr.SetI0(<double>value)
            
    @overrideE0.setter
    def overrideE0(self, value):  # @DuplicatedSignature
        self._thisptr.SetOverrideE0(<bool>value)
        
    @E0.setter
    def E0(self, value):  # @DuplicatedSignature
        self._thisptr.SetE0(<double complex>value)
        
    @mode.setter
    def mode(self, modeStr):  # @DuplicatedSignature
        self._thisptr.SetMode(NonlinearTmmModeFromStr(modeStr))
        

#===============================================================================
# SecondOrderNLIntensities
#===============================================================================

cdef class _SecondOrderNLIntensities:
    cdef readonly object P1, P2, Gen
    
    def __cinit__(self):
        pass
    
    cdef _Init(self, SecondOrderNLIntensitiesCpp* ptr):
        P1 = _Intensities()
        P1._Init(&ptr.P1)
    
        P2 = _Intensities()
        P2._Init(&ptr.P2)
        
        Gen = _Intensities()
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
    cdef readonly np.ndarray wlsGen, betasGen;
    
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
        self.wlsGen = ndarray_view(self._thisptr.wlsGen).squeeze()
        self.betasGen = ndarray_view(self._thisptr.betasGen).squeeze()
        
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
        self.P1.AddLayer(d, material)
        self.P2.AddLayer(d, material)
        self.Gen.AddLayer(d, material)
        
        # Cache material classes, avoids material dealloc
        self.materialsCache.append(material)
        
    def Solve(self):
        self._thisptr.Solve()
        
    def GetIntensities(self):
        cdef SecondOrderNLIntensitiesCpp resCpp;
        resCpp = self._thisptr.GetIntensities();
        
        res = _SecondOrderNLIntensities()
        res._Init(&resCpp)
        return res
        
    def Sweep(self, str paramStr, np.ndarray[double, ndim = 1] valuesP1, np.ndarray[double, ndim = 1] valuesP2, int layerNr = 0, double layerZ = 0.0, bool outPwr = True, bool outAbs = False, outEnh = False):
        cdef SweepResultSecondOrderNLTMMCpp *resCpp;
        cdef int outmask = 0
        if outPwr:
            outmask |= SWEEP_PWRFLOWS
        if outAbs:
            outmask |= SWEEP_ABS
        if outEnh:
            outmask |= SWEEP_ENH        
        
        cdef TMMParamCpp param;
        cdef int paramLayer = -1;
        if (paramStr.startswith("d_")):
            param = PARAM_LAYER_D
            paramLayer = int(paramStr[2:])
        else:
            param = TmmParamFromStr(paramStr)
        
        resCpp = self._thisptr.Sweep(TmmParamFromStr(paramStr), Map[ArrayXd](valuesP1), Map[ArrayXd](valuesP2), outmask, paramLayer, layerNr, layerZ)
        res = _SweepResultSecondOrderNLTMM()
        res._Init(resCpp);
        return res
    
    def WaveGetPowerFlows(self, int layerNr, double x0 = float("nan"), double x1 = float("nan"), double z = 0.0):
        cdef pair[double, double] res;
        res = self._thisptr.WaveGetPowerFlows(layerNr, x0, x1, z)
        return (res.first, res.second)
    
    def WaveGetFields2D(self, np.ndarray[double, ndim = 1] zs, np.ndarray[double, ndim = 1] xs, str dirStr = "total"):
        cdef FieldsZXCpp *resCpp;
        cdef WaveDirectionCpp direction = WaveDirectionFromStr(dirStr)
        resCpp = self._thisptr.WaveGetFields2D(Map[ArrayXd](zs), Map[ArrayXd](xs), direction)
        res = _FieldsZX()
        res._Init(resCpp)
        return res
        