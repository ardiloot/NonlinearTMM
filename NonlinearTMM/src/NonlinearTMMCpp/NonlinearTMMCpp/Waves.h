#pragma once
#include "Common.h"
#include "Material.h"
#include <unsupported/Eigen/FFT>

namespace TMM {

//---------------------------------------------------------------
// ENUMs
//---------------------------------------------------------------

enum class WaveType {
    PLANEWAVE,
    GAUSSIANWAVE,
    TUKEYWAVE,
    SPDCWAVE,
};

//---------------------------------------------------------------
// Functions
//---------------------------------------------------------------

[[nodiscard]] ArrayXd TukeyFunc(const ArrayXd& xs, double w0, double a);

//---------------------------------------------------------------
// Wave
//---------------------------------------------------------------

class Wave {
private:
    WaveType waveType;
    double pwr;
    bool overrideE0;
    double E0OverrideValue;
    double w0;
    Material* materialLayer0;
    Material* materialLayerThis;
    double Ly;
    double a; // Tukey param
    int nPointsInteg;
    double maxX;
    bool dynamicMaxX;
    double dynamicMaxXCoef;
    double dynamicMaxXAddition;
    double maxXThis;
    double maxPhi;
    double deltaKxSpdc; // SPDC
    bool solved;

    double wl;
    double beta;
    double k0;
    ArrayXd phis, kxs, kzs, fieldProfileXs, fieldProfile;
    ArrayXcd expansionCoefsKx;
    Eigen::FFT<double> fft;
    double E0;
    double k;
    double nLayer0;
    double thLayer0;
    double beamArea;

    void SolvePlaneWave();
    void SolveFFTWave();
    void SolveSpdcWave();

public:
    Wave();

    // Setters
    void SetWaveType(WaveType waveType_);
    void SetPwr(double pwr_);
    void SetOverrideE0(bool overrideE0_);
    void SetE0(double E0_);
    void SetW0(double w0_);
    void SetLy(double Ly_);
    void SetA(double a_);
    void SetNPointsInteg(int nPointsInteg_);
    void SetMaxX(double maxX_);
    void EnableDynamicMaxX(bool dynamicMaxX_);
    void SetDynamicMaxXCoef(double dynamicMaxXCoef_);
    void SetDynamicMaxXAddition(double dynamicMaxXAddition_);
    void SetMaxPhi(double maxPhi_);
    void SetParam(TMMParam param, double value);

    // Solve
    void Solve(double wl_, double beta_, Material* materialLayer0_, Material* materialLayerThis_,
               double deltaKxSpdc_ = constNAN);

    // Getters
    [[nodiscard]] WaveType GetWaveType() const noexcept;
    [[nodiscard]] double GetPwr() const noexcept;
    [[nodiscard]] bool GetOverrideE0() const noexcept;
    [[nodiscard]] double GetE0() const noexcept;
    [[nodiscard]] double GetW0() const noexcept;
    [[nodiscard]] double GetLy() const noexcept;
    [[nodiscard]] double GetA() const noexcept;
    [[nodiscard]] int GetNPointsInteg() const noexcept;
    [[nodiscard]] double GetMaxX() const noexcept;
    [[nodiscard]] bool IsDynamicMaxXEnabled() const noexcept;
    [[nodiscard]] double GetDynamicMaxXCoef() const noexcept;
    [[nodiscard]] double GetDynamicMaxXAddition() const noexcept;
    [[nodiscard]] double GetMaxPhi() const noexcept;
    [[nodiscard]] double GetDouble(TMMParam param) const;
    [[nodiscard]] pairdd GetXRange() const;
    [[nodiscard]] ArrayXd GetBetas() const;
    [[nodiscard]] ArrayXd GetPhis() const noexcept;
    [[nodiscard]] ArrayXd GetKxs() const;
    [[nodiscard]] ArrayXd GetKzs() const;
    [[nodiscard]] ArrayXd GetFieldProfileXs() const;
    [[nodiscard]] ArrayXd GetFieldProfile() const;
    [[nodiscard]] ArrayXcd GetExpansionCoefsKx() const;
    [[nodiscard]] double GetBeamArea() const;
    [[nodiscard]] bool IsCoherent() const noexcept;
};


} // namespace TMM
