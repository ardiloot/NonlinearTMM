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
    elif processStr == "spdc":
        return SPDC
    else:
        raise ValueError()

cdef WaveTypeCpp WaveTypeFromStr(str waveTypeStr):
    if waveTypeStr == "planewave":
        return PLANEWAVE
    if waveTypeStr == "gaussian":
        return GAUSSIANWAVE
    if waveTypeStr == "tukey":
        return TUKEYWAVE
    if waveTypeStr == "spdc":
        return SPDCWAVE;
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
    
cdef paramDictSecondOrderNLTMM = set(["deltaWlSpdc",
                                      "solidAngleSpdc",
                                      "deltaThetaSpdc"])
    
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
                "dynamicMaxXAddition",
                "maxPhi"])

    
cdef TMMParamCpp TmmParamFromStr(str paramStr):
    return paramDict[paramStr]
    
#===============================================================================
# Chi2Tensor
#===============================================================================

cdef class _Chi2Tensor:
    """_Chi2Tensor()
    
    This class is helper class for Material to accommodate the second-order
    susceptibility tensor. Allows setting nonlinearities by chi2- and by d-values.
    Possible to rotate initial tensor around all three axes.
    
    """
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
        """Update(**kwargs)
        
        This method changes the values of second-order nonlinearity tensor. By
        default the tensor is zero, not rotated and distinctFields set to True.
        
        Parameters
        ----------
        chiXYZ : float, optional
            Updates tensor value of second-order nonlinearity, where X, Y, Z
            denote numbers from 1..3 corresponding to x-, y-, z-axis.
        dXK : float, optional
            Updates tensor values by contracted notation, where dXK = 0.5 * chiXYZ
            and K is defined as: 1 = 11; 2 = 22; 3 = 33; 4 = 23, 32; 5 = 31, 13; 6 = 12, 21.
        distinctFields : bool, optional
            If set to True, then assumes that two input waves in nonlinear process
            are the same (e.g second harmonic generation). If False, then the
            input fields are not equal (as a result the result will be multiplied by 2).
            Default is true.
        phiX : float, optional
            The rotation angle (radians) around x-axis of the second order nonlinear tensor. 
        phiY : float, optional
            The rotation angle (radians) around y-axis of the second order nonlinear tensor. 
        phiZ : float, optional
            The rotation angle (radians) around z-axis of the second order nonlinear tensor.
            
        Examples
        --------
        Updates nonlinear tensor of material and then rotates it.
        
        >>> mat = Material.Static(1.5)
        >>> mat.chi2.Update(d11 = 1e-12, d22 = 2e-12, chi23 = 3e-12)
        >>> mat.chi2.Update(phiX = 0.2, phiY = 0.5, phiZ = -0.1)
                 
        """
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
            i1 = int(k[3])
            i2 = int(k[4])
            i3 = int(k[5])
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
        """
        Returns rotated 3x3x3 second-order nonlinearity tensor.
        
        Returns
        -------
        ndarray(3, 3, 3) of floats
        """
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
    """Material(wls, ns)
    
    This class describes the optical parameters of nonlinear medium. Default
    contructor takes arrays of wavelengths and complex refractive indices and
    does linear interpolation. For shortcut __call__ is defined as GetN.
    
    Parameters
    ----------
    wls : ndarray of floats
        Array of wavelengths (m)
    ns : ndarray of complex floats
        Corresponding complex refractive indices to the wls array.
    
    Attributes
    ----------
    chi2 : :class:`_Chi2Tensor`
        Instance of the :class:`_Chi2Tensor` helper class to store second-order nonlinearity tensor.
        
    """
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
        """GetN(wl)
        
        Returns refractive index of material at specified wavelength.
        
        Parameters
        ----------
        wl : float
            Wavelength in meters.
        
        Returns
        -------
        complex
            Complex refractive index at wavelength wl.
        
        Examples
        --------
        >>> wls = np.array([400e-9, 600e-9, 800e-9], dtype = float)
        >>> ns = np.array([1.5 + 0.3j, 1.7 + 0.2j, 1.8 + 0.1j], dtype = complex)
        >>> mat = Material(wls, ns)
        >>> mat(600e-9)
        1.7 + 0.2j
        
        """
        return self._thisptr.GetN(wl)
        
    def IsNonlinear(self):
        """IsNonlinear()
        
        This method returns True if material nonlinearity tensor :any:`chi2` is nonzero,
        False otherwise.
        
        Returns
        -------
        bool
            True if chi2 tensor is nonzero, False otherwise.
        """
        return self._thisptr.IsNonlinear()

#===============================================================================
# Waves
#===============================================================================

cdef class _Wave:
    """_Wave()
    
    This is a helper class for using non plane waves as an input.
        
    Attributes
    ----------
    waveType : str
        'planewave':
            Usual plane wave
        'gaussian':
            Standard Gaussian beam with waist size w0
        'tukey':
            Rectangular wave with cosine tapered edges (Tukey window). The width
            is given by :any:`w0` and the tapered region is determined by :any:`a`.
        'spdc':
            Wave used to represent vacuum fluctuations in case of SPDC calculation.
        Default is 'planewave'.
    pwr : float
        Power of the wave (in watts). Default 1.0. The area of the beam is
        calculated as w0 * Ly.
    overrideE0 : bool
        if True, then :any:`pwr` is ignored and :any:`E0` is used. Default False.
    E0 : float
        Maximum electrical field of the beam in vacuum.
    w0 : float
        Waist size of the beam (in meters). Dafualt is 0.1 mm.
    Ly : float
        Beam size in y-direction (in meters).
        Default is 1 mm.
    a : float
        Parameter used for Tukey wave. If a = 1, then the wave is perfect
        rectangle, if a = 0.5, then half of the profile is tapered by cosines.
        Defualt is 0.7.
    nPointsInteg : int
        Number of points used to represenbt the profile of the wave.
        Default is 100.
    maxX : float
        The wave profile is given in x-range (-maxX..maxX). Only used if
        dynamicMaxX is False. Default is 1 mm.
    dynamicMaxX : bool
        Selects between dynamic determination of x-span of the wave (region
        where :any:`nPointsInteg` samples are taken) or fixed span by :any:`maxX`.
        Default is True.
    dynamicMaxXCoef : float
        Dynamic maxX equals to the beam width (corrected to the angle of
        incidence) times :any:`dynamicMaxXCoef`. Default is 2.0.
    dynamicMaxXAddition : float
        Constant factor that is added to maxX if :any:`dynamicMaxX` is true.
        Default is 0.0.
    maxPhi : float
        Determines the maximum angle of deviation form the direction of
        propagation of the participating fields. Increases the range of x-span
        (i.e :any:`maxX`) if it is neccesary to limit the angular distribution of the
        participating plane waves. Default is 0.17 rad.
    xRange : tuple of floats (xMin, xMax)
        Current x-span calculated from :any:`maxX`, :any:`dynamicMaxXCoef` and :any:`maxPhi`.
        The wave must be solved to access this quantity.
    betas : ndarray of floats
        Normalized tangential wave vectors of plane wave expansion. The wave
        must be solved to access this quantity.
    phis : ndarray of floats
        Same information as in betas, but converted to angles in respect to the
        direction of the propagation. The wave must be solved to access this
        quantity.
    kxs : ndarray of floats
        Same information as in betas, but converted to the x-component of the
        wave vector. The wave must be solved to access this quantity.
    kzs : ndarray of floats
        Same information as in betas, but converted to the z-component of the
        wave vector. The wave must be solved to access this quantity.
    fieldProfile : tuple(2,) of ndarray of floats (xs, fiedProfile)
        The field profile of the input wave sample by :any:`nPointsInteg` points. The
        wave must be solved to access this quantity.
    expansionCoefsKx : ndarray of complex
        Array of plane wave expansion coefs.
                
    """
    
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
        """SetParams(**kwargs)
        
        Helper method to set all the attributes of the class.
        
        Returns 
        -------
        None
        
        """
        
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
    def dynamicMaxXAddition(self):
        return self._thisptr.GetDynamicMaxXAddition()

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

    @property
    def beamArea(self):
        return self._thisptr.GetBeamArea()


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
    
    @dynamicMaxXAddition.setter
    def dynamicMaxXAddition(self, double value): # @DuplicatedSignature
        self._thisptr.SetDynamicMaxXAddition(value)    

    @maxPhi.setter
    def maxPhi(self, double value): # @DuplicatedSignature
        self._thisptr.SetMaxPhi(value)    
     

#===============================================================================
# Intensities
#===============================================================================

cdef class _Intensities:
    """
    Helper class to store intensities for :class:`NonlinearTMM`.
    
    Attributes
    ----------
    inc : complex
        Electical field amplitute of the incident plane wave.
    r : complex
        Electical field amplitute of the reflected plane wave.
    t : complex
        Electrical field amplitute of the transmitted plane wave.
    I : float
        Intensity of the incident plane wave.
    R : float
        Intensity of the reflected plane wave.
    T : float
        Intensity of the transmitted plane wave.
        
    """
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
    """
    Helper class to store the result of the :any:`NonlinearTMM.Sweep` method.
    
    Attributes
    ----------
    inc : ndarray of complex
        The complex amplitutes of the incident plane waves.
    r : ndarray of complex
        The complex amplitutes of the reflected plane waves.
    t : ndarray of complex
        The complex amplitutes of the transmitted plane waves.
    Ii : ndarray of double
        The intensities of the incident plane waves.
    Ir : ndarray of double
        The intensities of the reflected plane waves.
    It : ndarray of double
        The intensities of the transmitted plane waves.
    Ia : ndarray of double
        The intensities of asborption in the structure.
    enh : ndarray of double
        The enhancment values of the electrical field norm.
        
    """
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
    """
    The helper class to store the results of `WaveSweep`.
    
    Attributes
    ----------
    Pi : ndarray of floats
        The power of the incident beam.
    Pr : ndarray of floats
        The power of the reflected beam.
    Pt : ndarray of floats
        The power of the transmitted beam.
    enh : ndarray of floats
        The enhancment of the beam.
    beamArea : ndarray of floats
        The area of the beam with correction of angle of incidence.
        
    """
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
    """
    Helper class to store the results of `GetFields`.
    
    Attributes
    ----------
    E : ndarray(N, 3) of complex
        Electrical fields along z-coordinate. First index is determines the
        z-coorinate and the second corrorrespond to the x-, y-, z-component
        (0..2).
    H : ndarray(N, 3) of complex
        Magnetic fields along z-coordinate. First index is determines the
        z-coorinate and the second corrorrespond to the x-, y-, z-component
        (0..2).
        
    """
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
    """
    Helper class to store the result of `GetFields2D` and `WaveGetFields2D`.
    
    Attributes
    ----------
    Ex : ndarray(N, M) of complex
        Electrical field x-component in zx-plane. First index corresponds to
        z-coorinate and the second to the x-coordinate.
    Ey : ndarray(N, M) of complex
        Electrical field y-component in zx-plane. First index corresponds to
        z-coorinate and the second to the x-coordinate.
    Ez : ndarray(N, M) of complex
        Electrical field z-component in zx-plane. First index corresponds to
        z-coorinate and the second to the x-coordinate.    
    EN : ndarray(N, M) of float
        Electical field norm in zx-plane. First index corresponds to
        z-coorinate and the second to the x-coordinate. 
    Hx : ndarray(N, M) of complex
        Magnetic field x-component in zx-plane. First index corresponds to
        z-coorinate and the second to the x-coordinate.    
    Hy : ndarray(N, M) of complex
        Magnetic field y-component in zx-plane. First index corresponds to
        z-coorinate and the second to the x-coordinate.
    Hz : ndarray(N, M) of complex
        Magnetic field z-component in zx-plane. First index corresponds to
        z-coorinate and the second to the x-coordinate.
    HN : ndarray(N, M) of float
        Magnetic field norm in zx-plane. First index corresponds to
        z-coorinate and the second to the x-coordinate.
          
    """
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
    """
    Helper class for `NonlinearLayer` to describe the parameters of the
    homogeneous waves in the layer.
    
    Attributes
    ----------
    kzF : complex
        Z-component of the wave vector.
    kx : float
        X-component of the wave vector.
    """
    cdef HomogeneousWaveCpp *_thisptr
    
    def __cinit__(self):
        self._thisptr = NULL
    
    cdef _Init(self, HomogeneousWaveCpp *ptr):
        self._thisptr = ptr
        
    def GetMainFields(self, double z):
        """GetMainFields(z)
        
        Returns forward and backward component of the main fields at the
        distance z.  In case of s-pol the main field is Ey and in case of p-pol
        the main field is Hy.
        
        Parameters
        ----------
        z : float 
            z-coorinate for the main fields calculation. z=0 is the beginning of
            the layer.
        
        Returns
        -------
        ndarray(2,) of complex
            The foreard and backward component of the main fields.
        
        """
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
    """
        Helper class for layer specific data and mathods.
        
        Attributes
        ----------
        d : float
            The thikness of the layer.
            
    """
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
        """GetIntensity(z)
        
        Returns intensity of the plane wave at coordinate `z`.
        
        Parameters
        ----------
        z : float
            z-coorinate
        
        Returns
        -------
        float
            The intensity of the plane waves in layer.
        
        """
        return self._thisptr.GetIntensity(z);
    
    def GetAbsorbedIntensity(self):
        """GetAbsorbedIntensity()
        
        Returns absorbed intensity in the layer.
        
        Returns
        -------
        float
        
        """
        return self._thisptr.GetAbsorbedIntensity();
    
    def GetSrcIntensity(self):
        """GetSrcIntensity()
        
        Calculates source power of the layer. Only nonzero if the material is
        nonlinear.
        
        Returns
        -------
        float
        
        """
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
    """NonlinearTMM()
    
    This class is mainly used to calculate linear propagation of plane waves
    in stratified medium. It can work like ordinary TMM and calculate the propagation
    of the input waves. It is also capable of nonlinear calculations, but
    for this purpose use specialized class :any:`SecondOrderNLTMM`.
    
    Default constructor takes no arguments.
    
    Attributes
    ----------
    E0 : complex
        The amplitude of the input plane wave (given in vaccuum, not in the first layer).
        Only used if :any:`overrideE0` is set to True (default is False), otherwise intensity
        :any:`I0` is used instead.
        Default: 1.0.
    I0 : float
        The intensity of the input beam in the first medium. Only used if
        :any:`overrideE0` is set to False (default).
        Default: 1.0.
    beta : float
        Normalized tangential wave vector. Determines the angle of incidence through
        relation beta = sin(theta) * n_p, where theta is angle of incidence and 
        n_p is the refractive index of the prism.
        Default: not defined.
    layers : list of helper class :any:`_NonlinerLayer`  
        Allows to access layers by index.
    mode : str
        By default "incident" which corresponds to Ordinary TMM. Other mode is, 
        "nonlinear" but it is automatically set by :any:`SecondOrderNLTMM` (i.e, this
        parameter has to be modified only in special cases).
    overrideE0 : bool
        If True, then parameter :any:`E0` is used, otherwise intensity :any:`I0` is used.
        Default: False.
    pol : str
        "p" or "s" corresponding to p- and s-polarization, respectively.
        Default: Not defined.
    wave : :any:`_Wave`
        Innstance of helper class :any:`_Wave` for non plane wave calculations.
    wl : float
        Wavelength of calculations in meters.
    
    """
    
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
        """AddLayer(d, material)
        
        Adds layer to the TMM.
        
        Parameters
        ----------
        d : float
            Thickness of the layer in meters. First and the last medium have
            infinite thickness.
        material : :any:`Material`
            Instance of the helper class :any:`Material`
        
        Examples
        --------
        >>> tmm = TMM()
        >>> tmm.AddLayer(float("inf"), Material.Static(1.5))
        >>> tmm.AddLayer(50e-9, Material.Static(0.3 + 3j))
        >>> tmm.AddLayer(float("inf"), Material.Static(1.0))
        """
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
        """SetParams(**kwargs)
        
        Helper method to set the values of all the attributes. See the docstring
        of :any:`NonlinearTMM`.
        
        """
        for name, value in kwargs.iteritems():
            if name not in paramDict:
                raise ValueError("Unknown kwarg %s" % (name))
            setattr(self, name, value)
        
    def Solve(self, **kwargs):
        """Solve(**kwargs)
        
        Solves the structure.
        
        Parameters
        ----------
        wl : float, optional
             Wavelength in meters.
        pol : str, optional
            "p" or "s" denoting the polarization of the waves.
        beta : float, optional
            Normalized tangential wave vector.
        E0 : complex, optional
            Input plane wave amplitude in vacuum. Only used if :any:`overrideE0` is
            True.
        I0 : float, optional
            Input plane wave intensity in the first layer. Only used if
            :any:`overrideE0` is False.
        overrideE0 : bool, optional
            Selects between :any:`E0` and :any:`I0`.
        
        Examples
        --------
        >>> tmm.Solve(wl = 532e-9, beta = 0.0)
            
        """
        self.SetParams(**kwargs)
        self._thisptr.Solve()
            
    def GetIntensities(self):
        """GetIntensities()
        
        Returns the intensities and amplitutes of incident, reflected and
        transmitted wave. The structure must be solved first.
        
        Returns
        -------
        :any:`_Intensities`
            Helper class to hold intensity data.
        """
        cdef IntensitiesCpp resCpp = self._thisptr.GetIntensities()
        res = _Intensities()
        res._Init(&resCpp)
        return res 
    
    def Sweep(self, str paramStr, np.ndarray[double, ndim = 1] values, int layerNr = 0, double layerZ = 0.0, bool outPwr = True, bool outAbs = False, outEnh = False):
        """Sweep(paramStr, values, layerNr = 0, layerZ = 0, outPwr = True, outAbs = False, outEnh = False)
        
        Solves the structure for series of :any:`values` of param :any:`paramStr`. Using
        this function is more confortable and faster than just changing params
        and solving the structure.
        
        Parameters
        ----------
        paramStr : str
            'wl':
                Wavelength in meters
            'beta':
                normalized tangential wavevector
            'I0':
                intensity of the wave
            'd_i': thikness of layer i (0..N-1)
        values : ndarray of floats
            Correspondig values of param :any:`paramStr`.
        layerNr : int
            Specifies layer, where electrical field enhancment is calculated.
        layerZ : double
            Specifies z-coordinate of enchncment calculation inside :any:`layerNr`.
        outPwr : bool
            Turns calculation of intensities on/off.
        outAbs : bool
            Turns calculation of absoprtiopn in the entire structure on/off.
        outEnh : bool
            Turns calculation of enhancment in layer :any:`layerNr` at distance
            :any:`layerZ` on/off.values
        
        Returns
        -------
        :any:`_SweepResultNonlinearTMM`
            Helper class to store the result.
            
        """
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
        """GetFields(zs, dir = "total")
        
        Calculates electical and magnetic fields along z-axis. The structure
        must be solved first.
        
        Parameters
        ----------
        zs : ndarray of floats
            Points on z-axis where to calculate the fields. The beginning of the
            first layer is at z = 0.
        dir : {'total', 'forward', 'backward'}
            Specifies the components of the output fields.
            
        Returns
        -------
        :any:`_FieldsZ`
            Helper class to store electric and magnetic fields.
            
        """
        cdef FieldsZCpp *resCpp;
        resCpp = self._thisptr.GetFields(Map[ArrayXd](zs), WaveDirectionFromStr(dir))
        res = _FieldsZ()
        res._Init(resCpp)
        return res
    
    def GetFields2D(self, np.ndarray[double, ndim = 1] zs, np.ndarray[double, ndim = 1] xs, str dir = "total"):
        """GetFields2D(zs, xs, dir = "total")
        
        Calculates electical and magnetic fields in xz-plane. Made as fast as
        possible. The structure must be solved first.
        
        Parameters
        ----------
        zs : ndarray of floats
            Points on z-axis where to calculate the fields. The beginning of the
            first layer is at z = 0.
        xs : ndarray of floats
            Points on x-axis where to calculate the fields.
        dir : {'total', 'forward', 'backward'}
            Specifies the components of the output fields.
            
        Returns
        -------
        :any:`_FieldsZX`
            Helper class to store electric and magnetic fields in regular grid.
            
        """
        cdef FieldsZXCpp *resCpp;
        resCpp = self._thisptr.GetFields2D(Map[ArrayXd](zs), Map[ArrayXd](xs), WaveDirectionFromStr(dir))
        res = _FieldsZX()
        res._Init(resCpp)
        return res
    
    def GetAbsorbedIntensity(self):
        """GetAbsorbedIntensity()
        
        Calculates intensity of absorption. The structure must be solved first.
        
        Returns
        -------
        float
            Absorption intensity
            
        """
        return self._thisptr.GetAbsorbedIntensity();
    
    def GetEnhancement(self, int layerNr, double z = 0.0):
        """GetEnhancement(layerNr, z = 0.0)
        
        Calculates the enhancement of electical field norm at specified layer at
        fixed z-cooridnate.
        
        Parameters
        ----------
        layerNr : int
            Number of layer where to calculate the enhancement.
        z : float
            Z-distance from the beginning of the layer.
            
        Returns
        -------
        float
            The enhancment of the electrical field norm in comparison to the
            input wave in the vacuum.
        
        """
        cdef double res
        res = self._thisptr.GetEnhancement(layerNr, z)
        return res
    
    # Waves
    
    def WaveGetPowerFlows(self, int layerNr, double x0 = float("nan"), double x1 = float("nan"), double z = 0.0):
        """WaveGetPowerFlows(layerNr, x0 = float("nan"), x1 = float("nan"), z = 0.0)
        
        Analogous to the :any:`GetIntensities`, but calculates the powers of the
        beams instead of the intensities of the plane-waves.
        
        Parameters
        ----------
        layerNr : int
            Specifies layer number where to calculate the power of the beam.
        x0 : float
            Specifies the starting point of the integration of the power in the
            x-direction. By default this parameter is selected by the :any:`_Wave`
            class.
        x1 : float
            Specifies the end point of the integration of the power in the
            x-direction. By default this parameter is selected by the :any:`_Wave`
            class.
        z : float
            Specifies the z-position of the line through which the integration
            of the power of the beam is done.
            
        Returns
        -------
        tuple of floats (PB, PF)
            PB is the power propageted into netagtive infinity and PF denotes
            the power propagation to the positive direction of z-axis.
            
        """
        # NonlinearLayer has its own specific method
        cdef pair[double, double] res;
        res = self._thisptr.WaveGetPowerFlows(layerNr, x0, x1, z)
        return (res.first, res.second)
    
    def WaveGetEnhancement(self, int layerNr, double z = 0.0):
        """WaveGetEnhancement(layerNr, z = 0.0)
        
        Calculates enhencment of electical field norm in comparison to the imput
        beam in vacuum. Analogous to :any:`GetEnhancement`.
        
        Parameters
        ----------
        layerNr : int
            Number of layer where to calculate the enhancement.
        z : float
            Z-distance from the beginning of the layer.
            
        Returns
        -------
        float
            The enhancment of the electrical field norm in comparison to the
            input wave in the vacuum.
        
        """
        cdef double res
        res = self._thisptr.WaveGetEnhancement(layerNr, z)
        return res
    
    def WaveSweep(self, str paramStr, np.ndarray[double, ndim = 1] values, \
            int layerNr = 0, double layerZ = 0.0, bool outPwr = True, \
            bool outR = False, bool outT = False, bool outEnh = False):
        """WaveSweep(paramStr, values, layerNr = 0, layerZ = 0.0, outPwr = True, outR = False, outT = False, outEnh = False)
        
        Solves the structure for waves for series of :any:`values` of param
        :any:`paramStr`. Using this function is more confortable and faster than just
        changing params and solving the structure. Analogous to :any:`Sweep`.
        
        Parameters
        ----------
        paramStr : str
            'wl':
                Wavelength in meters
            'beta':
                normalized tangential wavevector
            'I0':
                intensity of the wave
            'd_i':
                thikness of layer i (0..N-1)
            'w0':
                waist size of the input beam
        values : ndarray of floats
            Correspondig values of param :any:`paramStr`.
        layerNr : int
            Specifies layer, where electrical field enhancment is calculated.
        layerZ : double
            Specifies z-coordinate of enchncment calculation inside :any:`layerNr`.
        outPwr : bool
            Turns calculation of all powers on/off.
        outR : bool
            Turns calculation of reflected power on/off.
        outT : bool
            Turns calculation of transmitted power on/off.
        outEnh : bool
            Turns calculation of enhancment in layer layerNr at distance
            :any:`layerZ` on/off.
        
        Returns
        -------
        :any:`_SweepResultNonlinearTMM`
            Helper class to store the result.
            
        """
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
        """WaveGetFields2D(zs, xs, dirStr = "total")
        
        Calculates 2D electric and magnetic fields of beam propagating in the
        structure. Analogous to the :any:`GetFields2D` of plane waves.
        
        Parameters
        ----------
        zs : ndarray of floats
            Points on z-axis where to calculate the fields. The beginning of the
            first layer is at z = 0.
        xs : ndarray of floats
            Points on x-axis where to calculate the fields.
        dir : {'total', 'forward', 'backward'}
            Specifies the components of the output fields.
            
        Returns
        -------
        :any:`_FieldsZX`
            Helper class to store electric and magnetic fields in regular grid.
        
        """
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
    """_SecondOrderNLIntensities()
    This is a helper class for `SecondOrderNLTMM`. It stores the results of
    `GetIntensities`.
    
    Attributes
    ----------
    P1 : :any:`_Intensities`
        Instance of `_Intensities` class and holds the intensities of second
        inuput wave.
    P2 : :any:`_Intensities`
        Instance of `_Intensities` class and holds the intensities of first
        inuput wave.
    Gen : :any:`_Intensities`
        Instance of `_Intensities` class and holds the intensities of generated
        wave.
        
    """
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
    """_SweepResultSecondOrderNLTMM()
    Helper class for `SecondOrderNLTMM`. Stores the results of `Sweep` method.
    
    Attributes
    ----------
    P1 : :any:`_SweepResultNonlinearTMM`
        Sweep results for the first input wave.
    P2 : :any:`_SweepResultNonlinearTMM`
        Sweep results for the second input wave.
    Gen : :any:`_SweepResultNonlinearTMM`
        Sweep results for the generated wave.
    wlsGen : ndarray of floats
        Stores the wavelength of the generated wave.
    betasGen : ndarray of floats
        Stores the normalized tangential wave vector the generated wave.
        
    """
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
# WaveSweepResultSecondOrderNLTMM
#===============================================================================
    
cdef class _WaveSweepResultSecondOrderNLTMM:
    """_WaveSweepResultSecondOrderNLTMM()
    Helper class for `SecondOrderNLTMM`. Stores the results of `WaveSweep`
    method.
    
    Attributes
    ----------
    P1 : :any:`_WaveSweepResultNonlinearTMM`
        The sweep result of the first input beam. 
    P1 : :any:`_WaveSweepResultNonlinearTMM`
        The sweep result of the second input beam.
    Gen : :any:`_WaveSweepResultNonlinearTMM`
        The sweep result of the generated beam.
    wlsGen : ndarray of floats
        Stores the wavelength of the generated wave.
    betasGen : ndarray of floats
        Stores the normalized tangential wave vector the generated wave.
    
    """
    cdef WaveSweepResultSecondOrderNLTMMCpp *_thisptr
    cdef readonly object P1, P2, Gen
    cdef readonly np.ndarray wlsGen, betasGen;
    
    def __cinit__(self):
        self._thisptr = NULL
        pass
    
    def __dealloc__(self):
        if self._thisptr:
            del self._thisptr   

    cdef _Init(self, WaveSweepResultSecondOrderNLTMMCpp *ptr):
        self._thisptr = ptr
        
        P1 = _WaveSweepResultNonlinearTMM()
        P2 = _WaveSweepResultNonlinearTMM()
        Gen = _WaveSweepResultNonlinearTMM()
        
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
    """SecondOrderNLTMM(mode)
    
    This class calculates second-order nonliner processes (e.g. sum-frequency,
    difference frequency generation and SPDC) in layered structures. Relies on the
    functionality of `NonlinearTMM` class.
    
    Parameters
    ----------
    mode : str {'sfg', 'dfg', 'spdc'}
    
    Attributes
    ----------
    P1 : :any:`NonlinearTMM`
        The inctance of the first pump wave TMM.
    P2 : :any:`NonlinearTMM`
        The inctance of the second pump wave TMM.
    Gen : :any:`NonlinearTMM`
        The inctance of the generated wave TMM.
    deltaWlSpdc : float
        The spectral collection window (in nanometers) of the SPDC signal. 
        Only used if :any:`mode` is set to :any:`SPDC`
        Default: NaN.
    solidAngleSpdc : float
        The collection solid angle (srad) of the detector (given in vacuum). 
        Only used if :any:`mode` is set to :any:`SPDC`
        Default: NaN.
    deltaThetaSpdc : float
        The horizontal collection window (rad) of the detector (given in vacuum).
        It is used to calculate the vertical span of the detector. 
        Only used if :any:`mode` is set to :any:`SPDC`
        Default: NaN.
    
    """
    cdef SecondOrderNLTMMCpp *_thisptr
    cdef list materialsCache
    cdef readonly object P1, P2, Gen

    def __cinit__(self, str mode = "sfg", **kwargs):
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
        
        # Set params
        self.SetParams(**kwargs) 
    
    def __dealloc__(self):
        del self._thisptr
    
    # Methods
    #---------------------------------------------------------------------------
        
    def SetParams(self, **kwargs):
        """SetParams(**kwargs)
        
        Helper method to set the values of all the attributes. See the docstring
        of :any:`SecondOrderNLTMM`.
        
        """
        for name, value in kwargs.iteritems():
            if name not in paramDictSecondOrderNLTMM:
                raise ValueError("Unknown kwarg %s" % (name))
            setattr(self, name, value)
        
    def AddLayer(self, double d, Material material):
        """AddLayer(d, material)
        
        Adds layer to SecondOrderNLTMM. Equivalent of adding layer to P1, P2
        and Gen.
        
        Parameters
        ----------
        d : float
            layer thickness (m)
        material : :any:`Material`
            The class containing the material parameters.
            
        """
        # No copy of material is made
        self.P1.AddLayer(d, material)
        self.P2.AddLayer(d, material)
        self.Gen.AddLayer(d, material)
        
        # Cache material classes, avoids material dealloc
        self.materialsCache.append(material)
        
    def Solve(self):
        """Solve()
        
        Solves the structure.
            
        """
        self._thisptr.Solve()
        
    def UpdateGenParams(self):
        """UpdateGenParams()
        
        Forces the update of the wavelength and beta of the generated beam :any:`Gen`.

        """
        self._thisptr.UpdateGenParams()
        
    def GetIntensities(self):
        """GetIntensities()
        
        Returns the intensities and amplitutes of incident, reflected and
        transmitted wave for :any:`P1`, :any:`P2` and :any:`Gen`.
        The structure must be solved first.
        
        Returns
        -------
        :any:`_SecondOrderNLIntensities`
            Helper class to hold intensity data.
        """
        cdef SecondOrderNLIntensitiesCpp resCpp;
        resCpp = self._thisptr.GetIntensities();
        
        res = _SecondOrderNLIntensities()
        res._Init(&resCpp)
        return res
        
    def Sweep(self, str paramStr, np.ndarray[double, ndim = 1] valuesP1, np.ndarray[double, ndim = 1] valuesP2, int layerNr = 0, double layerZ = 0.0, bool outPwr = True, bool outAbs = False, bool outEnh = False, bool outP1 = True, bool outP2 = True, bool outGen = True):
        """Sweep(paramStr, valuesP1, valuesP2, layerNr = 0, layerZ = 0.0, outPwr = True, outAbs = False, outEnh = False, outP1 = True, outP2 = True, outGen = True)
        
        Solves the structure for series of :any:`valuesP1` and :any:`valuesP2`
        of param :any:`paramStr`. Using this function is more confortable and
        faster than just changing params and solving the structure (parallelized
        with OpenMP).
        
        Parameters
        ----------
        paramStr : str
            'wl':
                Wavelength in meters
            'beta':
                normalized tangential wavevector
            'I0':
                intensity of the wave
            'd_i': thikness of layer i (0..N-1)
        valuesP1 : ndarray of floats
            Correspondig values of param :any:`paramStr` for :any:`P1`.
        valuesP2 : ndarray of floats
            Correspondig values of param :any:`paramStr` for :any:`P2`.
        layerNr : int
            Specifies layer, where electrical field enhancment is calculated.
        layerZ : double
            Specifies z-coordinate of enchncment calculation inside :any:`layerNr`.
        outPwr : bool
            Turns calculation of intensities on/off.
        outAbs : bool
            Turns calculation of absoprtiopn in the entire structure on/off.
        outEnh : bool
            Turns calculation of enhancment in layer :any:`layerNr` at distance
            :any:`layerZ` on/off.values
        outP1 : bool
            Turns calculation of the :any:`P1` on/off.
        outP2 : bool
            Turns calculation of the :any:`P2` on/off.
        outGen : bool
            Turns calculation of the :any:`Gen` on/off.
        
        Returns
        -------
        :any:`_SweepResultSecondOrderNLTMM`
            Helper class to store the result.
            
        """
        cdef SweepResultSecondOrderNLTMMCpp *resCpp;
        cdef int outmask = 0
        if outP1:
            outmask |= SWEEP_P1
        if outP2:
            outmask |= SWEEP_P2
        if outGen:
            outmask |= SWEEP_GEN
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
        """WaveGetPowerFlows(layerNr, x0 = float("nan"), x1 = float("nan"), z = 0.0)
        
        Analogous to the :any:`GetIntensities`, but calculates the powers of the
        beams instead of the intensities of the plane-waves. Only for the
        calculation of the power of generated beam. For pump beams use the
        same method of :any:`NonlinearTMM`.
        
        Parameters
        ----------
        layerNr : int
            Specifies layer number where to calculate the power of the beam.
        x0 : float
            Specifies the starting point of the integration of the power in the
            x-direction. By default this parameter is selected by the :any:`_Wave`
            class.
        x1 : float
            Specifies the end point of the integration of the power in the
            x-direction. By default this parameter is selected by the :any:`_Wave`
            class.
        z : float
            Specifies the z-position of the line through which the integration
            of the power of the beam is done.
            
        Returns
        -------
        tuple of floats (PB, PF)
            PB is the power propageted into netagtive infinity and PF denotes
            the power propagation to the positive direction of z-axis.
            
        """
        cdef pair[double, double] res;
        res = self._thisptr.WaveGetPowerFlows(layerNr, x0, x1, z)
        return (res.first, res.second)
    
    def WaveSweep(self, str paramStr, np.ndarray[double, ndim = 1] valuesP1, \
            np.ndarray[double, ndim = 1] valuesP2,
            int layerNr = 0, double layerZ = 0.0, bool outPwr = True, \
            bool outR = False, bool outT = False, bool outEnh = False, \
            bool outP1 = True, bool outP2 = True, bool outGen = True):
        """WaveSweep(paramStr, valuesP1, valuesP2, layerNr = 0, layerZ = 0.0, outPwr = True, outR = False, outT = False, outEnh = False, outP1 = True, outP2 = True, outGen = True)
        
        Solves the structure for waves for series of :any:`valuesP1` and :any:`valuesP2` of param
        :any:`paramStr`. Using this function is more confortable and faster than just
        changing the params and solving the structure. Analogous to :any:`Sweep`.
        
        Parameters
        ----------
        paramStr : str
            'wl':
                Wavelength in meters
            'beta':
                normalized tangential wavevector
            'I0':
                intensity of the wave
            'd_i':
                thikness of layer i (0..N-1)
            'w0':
                waist size of the input beam
        valuesP1 : ndarray of floats
            Correspondig values of param :any:`paramStr` for :any:`P1`.
        valuesP2 : ndarray of floats
            Correspondig values of param :any:`paramStr` for :any:`P2`.
        layerNr : int
            Specifies layer, where electrical field enhancment is calculated.
        layerZ : double
            Specifies z-coordinate of enchncment calculation inside :any:`layerNr`.
        outPwr : bool
            Turns calculation of all powers on/off.
        outR : bool
            Turns calculation of reflected power on/off.
        outT : bool
            Turns calculation of transmitted power on/off.
        outEnh : bool
            Turns calculation of enhancment in layer layerNr at distance
            :any:`layerZ` on/off.
        outP1 : bool
            Turns calculation of the :any:`P1` on/off.
        outP2 : bool
            Turns calculation of the :any:`P2` on/off.
        outGen : bool
            Turns calculation of the :any:`Gen` on/off.
        
        Returns
        -------
        :any:`_WaveSweepResultSecondOrderNLTMM`
            Helper class to store the result.
            
        """
        
        cdef WaveSweepResultSecondOrderNLTMMCpp *resCpp;
        cdef int outmask = 0
        if outP1:
            outmask |= SWEEP_P1
        if outP2:
            outmask |= SWEEP_P2
        if outGen:
            outmask |= SWEEP_GEN
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
            
        resCpp = self._thisptr.WaveSweep(TmmParamFromStr(paramStr), Map[ArrayXd](valuesP1), Map[ArrayXd](valuesP2), outmask, paramLayer, layerNr, layerZ)
        res = _WaveSweepResultSecondOrderNLTMM()
        res._Init(resCpp);
        return res
    
    def WaveGetFields2D(self, np.ndarray[double, ndim = 1] zs, \
                        np.ndarray[double, ndim = 1] xs, str dirStr = "total"):
        """WaveGetFields2D(zs, xs, dirStr = "total")
        
        Calculates 2D electric and magnetic fields of beam propagating in the
        structure. Analogous to the :any:`GetFields2D` of plane waves. Only for the
        calculation of the power of generated beam. For pump beams use the
        same method of :any:`NonlinearTMM`.
        
        Parameters
        ----------
        zs : ndarray of floats
            Points on z-axis where to calculate the fields. The beginning of the
            first layer is at z = 0.
        xs : ndarray of floats
            Points on x-axis where to calculate the fields.
        dir : {'total', 'forward', 'backward'}
            Specifies the components of the output fields.
            
        Returns
        -------
        :any:`_FieldsZX`
            Helper class to store electric and magnetic fields in regular grid.
        
        """
        
        cdef FieldsZXCpp *resCpp;
        cdef WaveDirectionCpp direction = WaveDirectionFromStr(dirStr)
        resCpp = self._thisptr.WaveGetFields2D(Map[ArrayXd](zs), Map[ArrayXd](xs), direction)
        res = _FieldsZX()
        res._Init(resCpp)
        return res
    
    # Getter
    #---------------------------------------------------------------------------

    @property
    def deltaWlSpdc(self):
        return self._thisptr.GetDeltaWlSpdc()    

    @property
    def solidAngleSpdc(self):
        return self._thisptr.GetSolidAngleSpdc()  

    @property
    def deltaThetaSpdc(self):
        return self._thisptr.GetDeltaThetaSpdc()  
            
    # Setter
    #--------------------------------------------------------------------------- 
        
    @deltaWlSpdc.setter
    def deltaWlSpdc(self, value):  # @DuplicatedSignature
        self._thisptr.SetDeltaWlSpdc(<double>value)    

    @solidAngleSpdc.setter
    def solidAngleSpdc(self, value):  # @DuplicatedSignature
        self._thisptr.SetSolidAngleSpdc(<double>value)  

    @deltaThetaSpdc.setter
    def deltaThetaSpdc(self, value):  # @DuplicatedSignature
        self._thisptr.SetDeltaThetaSpdc(<double>value) 
        