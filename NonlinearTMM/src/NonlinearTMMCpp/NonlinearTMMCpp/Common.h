#pragma once
#define EIGEN_DONT_PARALLELIZE
#define NOMINMAX
#include <iostream>
#include <complex>
#include <algorithm>
#include <vector>
#include <memory>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#if defined(__SSE3__) || defined(_M_X64) || defined(_M_AMD64)
#include <pmmintrin.h>
#define TMM_USE_SSE3 1
#endif

namespace TMM {
using Eigen::Array2cd;
using Eigen::ArrayXcd;
using Eigen::ArrayXd;
using Eigen::Matrix2cd;
using Eigen::Matrix3d;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::Tensor;
using Eigen::Vector3cd;

//---------------------------------------------------------------
// Type aliases
//---------------------------------------------------------------

using dcomplex = std::complex<double>;
using Tensor3d = Tensor<double, 3>;
using pairdd = std::pair<double, double>;

//---------------------------------------------------------------
// Constants
//---------------------------------------------------------------

constexpr double PI = 3.141592653589793238462643383279502884L;
constexpr double constC = 299792458.0;
constexpr double constEps0 = 8.854187817e-12;
constexpr double constMu0 = 1.2566370614e-6;
inline constexpr dcomplex constI{0.0, 1.0};
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double constNAN = std::numeric_limits<double>::quiet_NaN();
constexpr double constHbar = 1.054571800e-34;

//---------------------------------------------------------------
// ENUMs
//---------------------------------------------------------------

enum class WaveDirection {
    F = 0,
    B = 1,
    TOT = -1,
};

enum class NonlinearProcess {
    SHG,
    SFG,
    DFG,
    SPDC,
    PROCESS_NOT_DEFINED,
};

enum class Polarization {
    P_POL,
    S_POL,
    NOT_DEFINED_POL,
};

enum class NonlinearTmmMode {
    MODE_INCIDENT,
    MODE_NONLINEAR,
    MODE_VACUUM_FLUCTUATIONS,
    MODE_NOT_DEFINED,
};

// Params that are used in sweep must be here
enum class TMMParam {
    PARAM_WL,
    PARAM_BETA,
    PARAM_POL,
    PARAM_I0,
    PARAM_OVERRIDE_E0,
    PARAM_E0,
    PARAM_MODE,
    PARAM_WAVE_W0,
    PARAM_LAYER_D,
    PARAM_NOT_DEFINED,
};

enum class TMMParamType { PTYPE_NONLINEAR_TMM, PTYPE_NONLINEAR_LAYER, PTYPE_WAVE };

// SweepOutput stays as plain enum for bitmask usage
enum SweepOutput {
    SWEEP_I = (1 << 0),
    SWEEP_R = (1 << 1),
    SWEEP_T = (1 << 2),
    SWEEP_ABS = (1 << 3),
    SWEEP_ENH = (1 << 4),
    SWEEP_P1 = (1 << 5),
    SWEEP_P2 = (1 << 6),
    SWEEP_GEN = (1 << 7),
    SWEEP_PWRFLOWS = (SWEEP_I | SWEEP_R | SWEEP_T),
    SWEEP_ALL_WAVES = (SWEEP_P1 | SWEEP_P2 | SWEEP_GEN),
    SWEEP_ALL_WAVE_PWRS = (SWEEP_PWRFLOWS | SWEEP_ALL_WAVES),
};

//---------------------------------------------------------------
// Cython compatibility: Cython's auto-generated integer-to-enum
// conversion code uses bitwise shift and OR operators. These must
// be provided for enum class types.
//---------------------------------------------------------------

#define ENUM_CLASS_CYTHON_COMPAT(EnumType)                                                                             \
    inline constexpr EnumType operator|(EnumType a, EnumType b) {                                                      \
        return static_cast<EnumType>(static_cast<int>(a) | static_cast<int>(b));                                       \
    }                                                                                                                  \
    inline constexpr EnumType operator<<(EnumType a, int shift) {                                                      \
        return static_cast<EnumType>(static_cast<int>(a) << shift);                                                    \
    }                                                                                                                  \
    inline constexpr EnumType operator*(EnumType a, EnumType b) {                                                      \
        return static_cast<EnumType>(static_cast<int>(a) * static_cast<int>(b));                                       \
    }

ENUM_CLASS_CYTHON_COMPAT(WaveDirection)
ENUM_CLASS_CYTHON_COMPAT(NonlinearProcess)
ENUM_CLASS_CYTHON_COMPAT(Polarization)
ENUM_CLASS_CYTHON_COMPAT(NonlinearTmmMode)
ENUM_CLASS_CYTHON_COMPAT(TMMParam)
ENUM_CLASS_CYTHON_COMPAT(TMMParamType)

#undef ENUM_CLASS_CYTHON_COMPAT

//---------------------------------------------------------------
// Functions
//---------------------------------------------------------------

[[nodiscard]] TMMParamType GetParamType(TMMParam param);
[[nodiscard]] constexpr double WlToOmega(double wl) {
    return 2.0 * PI * constC / wl;
}
[[nodiscard]] constexpr double OmegaToWl(double omega) {
    return 2.0 * PI * constC / omega;
}
[[nodiscard]] Matrix3d RotationMatrixX(double phi);
[[nodiscard]] Matrix3d RotationMatrixY(double phi);
[[nodiscard]] Matrix3d RotationMatrixZ(double phi);
[[nodiscard]] Tensor3d ApplyRotationMatrixToTensor(const Tensor3d& input, const Matrix3d& R);
[[nodiscard]] Tensor3d RotateTensor(Tensor3d& input, double phiX = 0, double phiY = 0, double phiZ = 0);
[[nodiscard]] constexpr double sqr(double a) {
    return a * a;
}
[[nodiscard]] inline dcomplex sqr(dcomplex a) {
    return a * a;
}
template <typename T>
[[nodiscard]] T Interpolate(double x, const ArrayXd& xs, const Eigen::Array<T, Eigen::Dynamic, 1>& ys);
[[nodiscard]] double GetDifferential(const ArrayXd& intVar, int nr);
[[nodiscard]] pairdd IntegrateWavePower([[maybe_unused]] int layerNr, Polarization pol, double wl, dcomplex epsLayer,
                                        const Eigen::MatrixX2cd& Us, const ArrayXd& kxs, const Eigen::MatrixX2cd& kzs,
                                        double x0, double x1, double z, double Ly);
[[nodiscard]] WaveDirection GetWaveDirection(dcomplex kzF, dcomplex eps, Polarization pol);
[[nodiscard]] ArrayXcd FFTShift(ArrayXcd data);
[[nodiscard]] ArrayXd IFFTShift(ArrayXd data);
[[nodiscard]] ArrayXd FFTFreq(int n, double dx);

//---------------------------------------------------------------
// Inline SSE
//---------------------------------------------------------------

[[nodiscard]] inline dcomplex FastExp(dcomplex z) {
    double c = std::exp(real(z));
    double y = imag(z);
    dcomplex res(c * std::cos(y), c * std::sin(y));
    return res;
}

[[nodiscard]] inline dcomplex multSSE(dcomplex aa, dcomplex bb) {
#ifdef TMM_USE_SSE3
    const __m128d mask = _mm_set_pd(-0.0, 0.0);

    // Load to registers
    __m128d a = _mm_load_pd(reinterpret_cast<const double*>(&aa));
    __m128d b = _mm_load_pd(reinterpret_cast<const double*>(&bb));

    // Real part
    __m128d ab = _mm_mul_pd(a, b);
    ab = _mm_xor_pd(ab, mask);

    // Imaginary part
    b = _mm_shuffle_pd(b, b, 1);
    b = _mm_mul_pd(b, a);

    // Combine
    ab = _mm_hadd_pd(ab, b);

    dcomplex res = 0.0;
    _mm_storeu_pd(reinterpret_cast<double*>(&res), ab);
    return res;
#else
    // Portable scalar fallback for non-x86 platforms (e.g., ARM)
    double ar = real(aa), ai = imag(aa);
    double br = real(bb), bi = imag(bb);
    return dcomplex(ar * br - ai * bi, ar * bi + ai * br);
#endif
}


} // namespace TMM
