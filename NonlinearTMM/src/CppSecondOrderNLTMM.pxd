from libcpp cimport bool
from eigency.core cimport *
from libcpp.pair cimport pair

#===============================================================================
# Common.h
#===============================================================================
        
cdef extern from "Common.h" namespace "TMM":
    cdef enum PolarizationCpp "TMM::Polarization":
        P_POL,
        S_POL
        
    cdef enum WaveDirectionCpp "TMM::WaveDirection":
        F,
        B,
        TOT
        
    cdef enum NonlinearTmmModeCpp "TMM::NonlinearTmmMode":
        MODE_INCIDENT,
        MODE_NONLINEAR
        
    cdef enum TMMParamCpp "TMM::TMMParam":
        PARAM_WL,
        PARAM_BETA,
        PARAM_POL,
        PARAM_I0,
        PARAM_OVERRIDE_E0,
        PARAM_E0,
        PARAM_MODE,
        PARAM_WAVE_W0,
        PARAM_LAYER_D,
        
    cdef enum NonlinearProcessCpp "TMM::NonlinearProcess":
        SFG,
        DFG,
        SPDC,
        
    cdef enum SweepOutputCpp "TMM::SweepOutput":
        SWEEP_PWRFLOWS,
        SWEEP_ABS,
        SWEEP_ENH,
        SWEEP_I,
        SWEEP_R,
        SWEEP_T,
        SWEEP_P1,
        SWEEP_P2,
        SWEEP_GEN,
        SWEEP_ALL_WAVES,
        SWEEP_ALL_WAVE_PWRS,

#===============================================================================
# Material.h
#===============================================================================

cdef extern from "Material.h" namespace "TMM":

    cdef cppclass Chi2TensorCpp "TMM::Chi2Tensor":
        void SetDistinctFields(bool distinctFields_) except +
        void SetRotation(double phiX_, double phiY_, double phiZ_) except +
        void SetChi2(int i1, int i2, int i3, double value) except +
        void SetD(int i1, int i2, double value) except +
        void Clear() except +
        double GetChi2Element(int i1, int i2, int i3) except +
    
    cdef cppclass MaterialCpp "TMM::Material":
        Chi2TensorCpp chi2
        MaterialCpp(Map[ArrayXd] & wlsExp, Map[ArrayXcd] & nsExp) except +
        double complex GetN(double wl) except +
        bool IsNonlinear() except +

#===============================================================================
# Waves.h
#===============================================================================

cdef extern from "Waves.h" namespace "TMM":
    cdef enum WaveTypeCpp "TMM::WaveType":
        PLANEWAVE,
        GAUSSIANWAVE,
        TUKEYWAVE,
        SPDCWAVE

    cdef cppclass WaveCpp "TMM::Wave":
        void SetWaveType(WaveTypeCpp waveType_) except +
        void SetPwr(double pwr_) except +
        void SetOverrideE0(bool overrideE0_) except +
        void SetE0(double E0_) except +
        void SetW0(double w0_) except +
        void SetMaterial(MaterialCpp *material_) except +
        void SetLy(double Ly_) except +
        void SetA(double a_) except +
        void SetNPointsInteg(int nPointsInteg_) except +
        void SetMaxX(double maxX_) except +
        void EnableDynamicMaxX(bool dynamicMaxX_) except +
        void SetDynamicMaxXCoef(double dynamicMaxXCoef_) except +
        void SetDynamicMaxXAddition(double) except +
        void SetMaxPhi(double maxPhi_) except +
        
        double GetPwr() except +
        bool GetOverrideE0() except +
        double GetE0() except +
        double GetW0() except +
        double GetLy() except +
        double GetA() except +
        int GetNPointsInteg() except +
        double GetMaxX() except +
        bool IsDynamicMaxXEnabled() except +
        double GetDynamicMaxXCoef() except +
        double GetDynamicMaxXAddition() except +
        double GetMaxPhi() except +
        pair[double, double] GetXRange() except +
        ArrayXd GetBetas() except +
        ArrayXd GetPhis() except +
        ArrayXd GetKxs() except +
        ArrayXd GetKzs() except +
        ArrayXd GetFieldProfileXs() except +
        ArrayXd GetFieldProfile() except +
        double GetBeamArea() except +
        ArrayXcd GetExpansionCoefsKx() except +
        
#===============================================================================
# NonlinearLayer.h
#===============================================================================

cdef extern from "NonlinearLayer.h" namespace "TMM":

    cdef cppclass HomogeneousWaveCpp "TMM::HomogeneousWave":
        double complex GetKzF() except +
        double GetKx() except +
        Array2cd GetMainFields(double z) except +

    cdef cppclass NonlinearLayerCpp "TMM::NonlinearLayer":
        HomogeneousWaveCpp* GetHw() except +
        void SetThickness(double d_) except +
        double GetThickness() except +
        double GetIntensity(double z) except +
        double GetAbsorbedIntensity() except +
        double GetSrcIntensity() except +
        
#===============================================================================
# NonlinearTMM.h
#===============================================================================

cdef extern from "NonlinearTMM.h" namespace "TMM":

    cdef cppclass IntensitiesCpp "TMM::Intensities":
        double complex inc, r, t
        double I, R, T
        
    cdef cppclass SweepResultNonlinearTMMCpp "TMM::SweepResultNonlinearTMM":
        ArrayXcd inc, r, t;
        ArrayXd Ii, Ir, It, Ia, enh;
        
        int GetOutmask();

    cdef cppclass WaveSweepResultNonlinearTMMCpp "TMM::WaveSweepResultNonlinearTMM":
        ArrayXd Pi, Pr, Pt, enh, beamArea;
        
        int GetOutmask();

    cdef cppclass FieldsZCpp "TMM::FieldsZ":
        MatrixXcd E, H;

    cdef cppclass FieldsZXCpp "TMM::FieldsZX":
        MatrixXcd Ex, Ey, Ez, Hx, Hy, Hz;
        PolarizationCpp GetPol()

    cdef cppclass NonlinearTMMCpp "TMM::NonlinearTMM":
        NonlinearTMMCpp() except +
        void AddLayer(double d_, MaterialCpp *material_) except +
        NonlinearLayerCpp* GetLayer(int layerNr)  except +
        int LayersCount() except +
       
        void SetWl(double wl_) except +
        void SetBeta(double beta_) except +
        void SetPolarization(PolarizationCpp pol_) except +
        void SetI0(double I0_) except +
        void SetOverrideE0(bool overrideE0_) except +
        void SetE0(double complex E0_) except +
        void SetMode(NonlinearTmmModeCpp mode_) except +
        void SetParam(TMMParamCpp param, double value) except +
        void SetParam(TMMParamCpp param, double complex value) except +
        
        double GetWl() except +
        double GetBeta() except +
        PolarizationCpp GetPolarization() except +
        double GetI0() except +
        bool GetOverrideE0() except +
        double complex GetE0() except +
        NonlinearTmmModeCpp GetMode() except +
        double GetDouble(TMMParamCpp param) except +
        complex GetComplex(TMMParamCpp param) except +
       
        # Plane waves
        void Solve() except +
        IntensitiesCpp GetIntensities() except +
        SweepResultNonlinearTMMCpp* Sweep(TMMParamCpp param, Map[ArrayXd] &, int outmask, int paramLayer, int layerNr, double layerZ) except +
        FieldsZCpp* GetFields(Map[ArrayXd] &, WaveDirectionCpp dir) except +
        FieldsZXCpp* GetFields2D(Map[ArrayXd] &, Map[ArrayXd] &, WaveDirectionCpp dir) except +
        double GetAbsorbedIntensity() except +
        double GetEnhancement(int layerNr, double z) except +
        
        # Waves
        WaveCpp* GetWave() except +
        pair[double, double] WaveGetPowerFlows(int layerNr, double x0, double x1, double z) except +
        double WaveGetEnhancement(int layerNr, double z) except +
        WaveSweepResultNonlinearTMMCpp* WaveSweep(TMMParamCpp param, Map[ArrayXd] &, int outmask, int paramLayer, int layerNr, double layerZ) except +
        FieldsZXCpp* WaveGetFields2D(Map[ArrayXd] &, Map[ArrayXd] &, WaveDirectionCpp) except +
      
        
#===============================================================================
# SecondOrderNLTMM.h
#===============================================================================

cdef extern from "SecondOrderNLTMM.h" namespace "TMM":

    cdef cppclass SecondOrderNLIntensitiesCpp "TMM::SecondOrderNLIntensities":
        IntensitiesCpp P1, P2, Gen;
        
    cdef cppclass SweepResultSecondOrderNLTMMCpp "TMM::SweepResultSecondOrderNLTMM":
        SweepResultNonlinearTMMCpp P1, P2, Gen;
        ArrayXd wlsGen, betasGen;
    
    cdef cppclass WaveSweepResultSecondOrderNLTMMCpp "TMM::WaveSweepResultSecondOrderNLTMM":
        WaveSweepResultNonlinearTMMCpp P1, P2, Gen;
        ArrayXd wlsGen, betasGen;
        
    cdef cppclass SecondOrderNLTMMCpp "TMM::SecondOrderNLTMM":
        SecondOrderNLTMM() except +
        void SetProcess(NonlinearProcessCpp process_) except +
        void SetDeltaWlSpdc(double value) except +
        void SetSolidAngleSpdc(double value) except +
        void SetDeltaThetaSpdc(double value) except +
        void AddLayer(double d_, MaterialCpp *material_) except +
        NonlinearTMMCpp* GetP1() except +
        NonlinearTMMCpp* GetP2() except +
        NonlinearTMMCpp* GetGen() except +
        double GetDeltaWlSpdc() except +
        double GetSolidAngleSpdc() except +
        double GetDeltaThetaSpdc() except +
        void UpdateGenParams() except +
        
        void Solve() except +
        SecondOrderNLIntensitiesCpp GetIntensities() except +
        SweepResultSecondOrderNLTMMCpp* Sweep(TMMParamCpp param, Map[ArrayXd] &, Map[ArrayXd] &, int outmask, int paramLayer, int layerNr, double layerZ) except +
        
        pair[double, double] WaveGetPowerFlows(int layerNr, double x0, double x1, double z) except +
        WaveSweepResultSecondOrderNLTMMCpp * WaveSweep(TMMParamCpp param, Map[ArrayXd] & valuesP1, Map[ArrayXd] & valuesP2, int outmask, int paramLayer, int layerNr, double layerZ) except +
        FieldsZXCpp * WaveGetFields2D(Map[ArrayXd]& zs, Map[ArrayXd]& xs, WaveDirectionCpp dir) except +
