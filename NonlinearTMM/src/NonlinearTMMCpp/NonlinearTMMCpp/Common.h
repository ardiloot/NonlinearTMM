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
		MODE_NOT_DEFINED,
	};

	enum TMMParam {
		PARAM_WL,
		PARAM_BETA,
		PARAM_POL,
		PARAM_I0,
		PARAM_OVERRIDE_E0,
		PARAM_E0,
		PARAM_MODE,
		PARAM_NOT_DEFINED,
	};

	enum SweepOutput {
		SWEEP_PWRFLOWS = (1 << 0),
		SWEEP_ABS = (1 << 1),
		SWEEP_ENH = (1 << 2),
	};

	//---------------------------------------------------------------
	// Functions
	//---------------------------------------------------------------

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
	double const IntegrateWavePower(int layerNr, Polarization pol, double wl, dcomplex epsLayer0, const ArrayXcd &Us,
		const ArrayXd &kxs, const ArrayXcd &kzs,
		double x0, double x1, double z, double Ly);
	WaveDirection GetWaveDirection(dcomplex kzF, dcomplex eps, Polarization pol);
	ArrayXcd FFTShift(ArrayXcd data);
	ArrayXd IFFTShift(ArrayXd data);
	ArrayXd FFTFreq(int n, double dx);

}