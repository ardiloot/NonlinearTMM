#pragma once
#define EIGEN_DONT_PARALLELIZE
#include <iostream>
#include <complex>
#include <algorithm>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
//#include <omp.h>

namespace TMM {
	using Eigen::Array2cd;
	using Eigen::ArrayXd;
	using Eigen::ArrayXcd;
	using Eigen::Vector3cd;
	using Eigen::Matrix2cd;
	using Eigen::Matrix3d;
	using Eigen::MatrixXd;
	using Eigen::MatrixXcd;
	using Eigen::Tensor;
	
	//---------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------

	typedef std::complex<double> dcomplex;
	typedef Tensor<double, 3> Tensor3d;
	typedef std::pair<double, double> pairdd;

	//---------------------------------------------------------------
	// Constants
	//---------------------------------------------------------------

	const double PI = 3.141592653589793238462643383279502884L;
	const double constC = 299792458.0;
	const double constEps0 = 8.854187817e-12;
	const double constMu0 = 1.2566370614e-6;
	const dcomplex constI = dcomplex(0.0, 1.0);
	const double INF = std::numeric_limits<double>::infinity();
	const double constNAN = std::numeric_limits<double>::quiet_NaN();
	const double constHbar = 1.054571800e-34;

	//---------------------------------------------------------------
	// ENUMs
	//---------------------------------------------------------------

	enum WaveDirection {
		F,
		B,
		TOT = -1,
	};

	enum NonlinearProcess {
		SHG,
		SFG,
		DFG,
		SPDC,
		PROCESS_NOT_DEFINED,
	};

	enum Polarization {
		P_POL,
		S_POL,
		NOT_DEFINED_POL,
	};

	enum NonlinearTmmMode {
		MODE_INCIDENT,
		MODE_NONLINEAR,
		MODE_VACUUM_FLUCTUATIONS,
		MODE_NOT_DEFINED,
	};

	// Params that are used in sweep must be here
	enum TMMParam {
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

	enum TMMParamType {
		PTYPE_NONLINEAR_TMM,
		PTYPE_NONLINEAR_LAYER,
		PTYPE_WAVE
	};

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
	// Functions
	//---------------------------------------------------------------

	TMMParamType GetParamType(TMMParam param);
	double WlToOmega(double wl);
	double OmegaToWl(double omega);
	Matrix3d RotationMatrixX(double phi);
	Matrix3d RotationMatrixY(double phi);
	Matrix3d RotationMatrixZ(double phi);
	Tensor3d ApplyRotationMatrixToTensor(Tensor3d input, Matrix3d R);
	Tensor3d RotateTensor(Tensor3d &input, double phiX = 0, double phiY = 0, double phiZ = 0);
	double sqr(double a);
	dcomplex sqr(dcomplex a);
	template <typename T> T Interpolate(double x, const ArrayXd & xs, const Eigen::Array<T, Eigen::Dynamic, 1> & ys);
	double GetDifferential(const ArrayXd &intVar, int nr);
	pairdd const IntegrateWavePower(int layerNr, Polarization pol, double wl, dcomplex epsLayer, const Eigen::MatrixX2cd &Us,
		const ArrayXd &kxs, const Eigen::MatrixX2cd &kzs,
		double x0, double x1, double z, double Ly);
	WaveDirection GetWaveDirection(dcomplex kzF, dcomplex eps, Polarization pol);
	ArrayXcd FFTShift(ArrayXcd data);
	ArrayXd IFFTShift(ArrayXd data);
	ArrayXd FFTFreq(int n, double dx);

	//---------------------------------------------------------------
	// Inline SSE
	//---------------------------------------------------------------

	__forceinline dcomplex FastExp(dcomplex z) {
		/*
		double y = imag(z);
		__m128d c2 = _mm_set1_pd(std::exp(real(z)));
		__m128d xy = _mm_set_pd(std::cos(y), std::sin(y));
		xy = _mm_mul_pd(xy, c2);
		dcomplex res;
		_mm_storeu_pd((double*)&(res), xy);
		*/
		
		double c = std::exp(real(z));
		double y = imag(z);
		dcomplex res(c * std::cos(y), c * std::sin(y));
		return res;
	}

	__forceinline dcomplex multSSE(dcomplex aa, dcomplex bb) {
		const __m128d mask = _mm_set_pd(-0.0, 0.0);

		// Load to registers
		__m128d a = _mm_load_pd((double*)&aa);
		__m128d b = _mm_load_pd((double*)&bb);

		// Real part
		__m128d ab = _mm_mul_pd(a, b);
		ab = _mm_xor_pd(ab, mask);

		// Imaginary part
		b = _mm_shuffle_pd(b, b, 1);
		b = _mm_mul_pd(b, a);

		// Combine
		ab = _mm_hadd_pd(ab, b);

		dcomplex res = 0.0;
		_mm_storeu_pd((double*)&(res), ab);
		return res;
	}


}