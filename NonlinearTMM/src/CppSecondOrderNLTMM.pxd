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
        
    cdef enum NonlinearProcessCpp "TMM::NonlinearProcess":
        SFG,
        DFG,
        
    cdef enum SweepOutputCpp "TMM::SweepOutput":
        SWEEP_PWRFLOWS,
        SWEEP_ABS,
        SWEEP_ENH,

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
        double GetPowerFlow(double z) except +
        double GetAbsorbedPower() except +
        double GetSrcPower() except +
        
#===============================================================================
# NonlinearTMM.h
#===============================================================================

cdef extern from "NonlinearTMM.h" namespace "TMM":

    cdef cppclass PowerFlowsCpp "TMM::PowerFlows":
        double complex inc, r, t
        double I, R, T
        
    cdef cppclass SweepResultNonlinearTMMCpp "TMM::SweepResultNonlinearTMM":
        ArrayXcd inc, r, t;
        ArrayXd I, R, T, A, enh;
        
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
        void Solve() except +
        PowerFlowsCpp GetPowerFlows() except +
        SweepResultNonlinearTMMCpp* Sweep(TMMParamCpp param, Map[ArrayXd] &, int outmask, int layerNr, double layerZ) except +
        FieldsZCpp* GetFields(Map[ArrayXd] &, WaveDirectionCpp dir) except +
        FieldsZXCpp* GetFields2D(Map[ArrayXd] &, Map[ArrayXd] &, WaveDirectionCpp dir) except +
        FieldsZXCpp* GetWaveFields2D(Map[ArrayXd] &, Map[ArrayXcd] &, Map[ArrayXd] &, Map[ArrayXd] &, WaveDirectionCpp) except +
        double GetAbsorbedPower() except +
        pair[double, double] GetPowerFlowsForWave(Map[ArrayXd] &betas, Map[ArrayXcd] &E0s, int layerNr, double x0, double x1, double z, double Ly, WaveDirectionCpp dir) except +

        void SetParam(TMMParamCpp param, bool value) except +
        void SetParam(TMMParamCpp param, int value) except +
        void SetParam(TMMParamCpp param, double value) except +
        void SetParam(TMMParamCpp param, double complex value) except +
        
        bool GetBool(TMMParamCpp param) except +
        int GetInt(TMMParamCpp param) except +
        double GetDouble(TMMParamCpp param) except +
        complex GetComplex(TMMParamCpp param) except +
        
#===============================================================================
# SecondOrderNLTMM.h
#===============================================================================

cdef extern from "SecondOrderNLTMM.h" namespace "TMM":

    cdef cppclass SecondOrderNLPowerFlowsCpp "TMM::SecondOrderNLPowerFlows":
        PowerFlowsCpp P1, P2, Gen;
        
    cdef cppclass SweepResultSecondOrderNLTMMCpp "TMM::SweepResultSecondOrderNLTMM":
        SweepResultNonlinearTMMCpp P1, P2, Gen;
        
    cdef cppclass SecondOrderNLTMMCpp "TMM::SecondOrderNLTMM":
        SecondOrderNLTMM() except +
        void SetProcess(NonlinearProcessCpp process_) except +
        void AddLayer(double d_, MaterialCpp *material_) except +
        void Solve() except +
        SecondOrderNLPowerFlowsCpp GetPowerFlows() except +
        NonlinearTMMCpp* GetP1() except +
        NonlinearTMMCpp* GetP2() except +
        NonlinearTMMCpp* GetGen() except +
        SweepResultSecondOrderNLTMMCpp* Sweep(TMMParamCpp param, Map[ArrayXd] &, Map[ArrayXd] &, int outmask, int layerNr, double layerZ) except +
        FieldsZXCpp * GetGenWaveFields2D(Map[ArrayXd]& betasP1, Map[ArrayXd]& betasP2, Map[ArrayXcd]& E0sP1, Map[ArrayXcd]& E0sP2,Map[ArrayXd]& zs, Map[ArrayXd]& xs, WaveDirectionCpp dir) except +
        pair[double, double] GetPowerFlowsGenForWave(Map[ArrayXd]& betasP1, Map[ArrayXd]& betasP2, Map[ArrayXcd]& E0sP1, Map[ArrayXcd]& E0sP2, int layerNr, double x0, double x1, double z, double Ly, WaveDirectionCpp dir) except +
    