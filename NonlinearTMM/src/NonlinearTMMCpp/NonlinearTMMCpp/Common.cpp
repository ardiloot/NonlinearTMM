#include "Common.h"

namespace TMM {
	TMMParamType GetParamType(TMMParam param) {
		switch (param)
		{
		case TMM::PARAM_WL:
		case TMM::PARAM_BETA:
		case TMM::PARAM_POL:
		case TMM::PARAM_I0:
		case TMM::PARAM_OVERRIDE_E0:
		case TMM::PARAM_E0:
		case TMM::PARAM_MODE:
			return PTYPE_NONLINEAR_TMM;
			break;
		case TMM::PARAM_WAVE_W0:
			return PTYPE_WAVE;
			break;
		case TMM::PARAM_LAYER_D:
			return PTYPE_NONLINEAR_LAYER;
			break;
		default:
			std::cerr << "Param has no type" << std::endl;
			throw std::invalid_argument("Param has no type");
			break;
		}
	}
	double WlToOmega(double wl) {
		double omega = 2.0 * PI * constC / wl;
		return omega;
	}

	double OmegaToWl(double omega)
	{
		double wl = 2.0 * PI * constC / omega;
		return wl;
	}

	Matrix3d RotationMatrixX(double phi) {
		Matrix3d res;
		res << 1.0, 0.0, 0.0, 0.0, std::cos(phi), -std::sin(phi), 0.0, std::sin(phi), std::cos(phi);
		return res;
	}

	Matrix3d RotationMatrixY(double phi) {
		Matrix3d res;
		res << std::cos(phi), 0.0, std::sin(phi), 0.0, 1.0, 0.0, -std::sin(phi), 0.0, std::cos(phi);
		return res;
	}

	Matrix3d RotationMatrixZ(double phi) {
		Matrix3d res;
		res << std::cos(phi), -std::sin(phi), 0.0, std::sin(phi), std::cos(phi), 0.0, 0.0, 0.0, 1.0;
		return res;
	}

	Tensor3d ApplyRotationMatrixToTensor(Tensor3d input, Matrix3d R) {
		Tensor3d res(3, 3, 3);
		res.setZero();

		for (int i1 = 0; i1 < 3; i1++) {
			for (int i2 = 0; i2 < 3; i2++) {
				for (int i3 = 0; i3 < 3; i3++) {
					for (int j1 = 0; j1 < 3; j1++) {
						for (int j2 = 0; j2 < 3; j2++) {
							for (int j3 = 0; j3 < 3; j3++) {
								res(i1, i2, i3) += R(i1, j1) * R(i2, j2) * R(i3, j3) * input(j1, j2, j3);
							}
						}
					}
				}
			}
		}
		return res;
	}

	Tensor3d RotateTensor(Tensor3d & input, double phiX, double phiY, double phiZ)
	{
		Tensor3d output(3, 3, 3);
		output = input;

		Matrix3d Rx = RotationMatrixX(phiX);
		Matrix3d Ry = RotationMatrixY(phiY);
		Matrix3d Rz = RotationMatrixZ(phiZ);

		output = ApplyRotationMatrixToTensor(output, Rx);
		output = ApplyRotationMatrixToTensor(output, Ry);
		output = ApplyRotationMatrixToTensor(output, Rz);
		return output;
	}

	double sqr(double a) {
		return a * a;
	}

	dcomplex sqr(dcomplex a) {
		return a * a;
	}

	template<typename T> T Interpolate(double x, const ArrayXd & xs, const Eigen::Array<T, Eigen::Dynamic, 1> & ys) {
		// xs must be sorted

		// Check range
		if (x < xs(0) || x >= xs(xs.size() - 1)) {
			throw std::runtime_error("Interpolation out of range");
		}

		if (xs(0) >= xs(xs.size() - 1)) {
			throw std::runtime_error("Interpolation: xs must be sorted");
		}

		// Binary search (last element, that is less or equal than x)
		int b = 0, e = xs.size() - 1;
		while (b < e) {
			int a = (b + e) / 2;
			if (xs(b) >= xs(e)) {
				throw std::runtime_error("Interpolation: xs must be sorted");
			}

			if (xs(a) > x) {
				// [b..a[
				e = a - 1;

				if (xs(e) <= x) {
					b = e;
				}
			}
			else {
				// [a..e]
				b = a; 
				if (xs(a + 1) > x) {
					e = a;
				}
			}
		}
		// Linear interpolation in range x[b]..x[b+1]
		double dx = xs(b + 1) - xs(b);
		T dy = ys(b + 1) - ys(b);
		T res = ys(b) + dy / dx * (x - xs(b));
		return res;
	}

	template dcomplex Interpolate<dcomplex>(double, const ArrayXd &, const ArrayXcd &);
	template double Interpolate<double>(double, const ArrayXd &, const ArrayXd &);

	double GetDifferential(const ArrayXd & intVar, int nr) {
		double dIntVar;
		if (intVar.size() <= 1) {
			dIntVar = 1.0;
		}
		else if (nr == 0) {
			dIntVar = intVar(1) - intVar(0);
		}
		else if (nr + 1 == intVar.size()) {
			dIntVar = intVar(intVar.size() - 1) - intVar(intVar.size() - 2);
		}
		else {
			dIntVar = 0.5 * (intVar(nr + 1) - intVar(nr - 1));
		}
		return dIntVar;
	}

	/*
	inline dcomplex CalcFx(dcomplex dk, double x0, double x1) {
		dcomplex exp1 = FastExp(constI * dk * x1);
		dcomplex exp2 = FastExp(constI * dk * x0);
		dcomplex res  = -constI * (exp1 - exp2) / dk;
		return res;
	}
	*/

	inline double CalcFxSym(double dk, double x0) {
		double res = -2.0 * std::sin(x0 * dk) / dk;
		return res;
	}
	
	/*
	double const IntegrateWavePower(int layerNr, Polarization pol, double wl, dcomplex epsLayer0, const ArrayXcd & Us, const ArrayXd & kxs, const ArrayXcd & kzs, double x0, double x1, double z, double Ly) {
		
		if (x0 != -x1) {
			std::cerr << "Currently only x0 = -x1 supported." << std::endl;
			throw std::invalid_argument("Currently only x0 = -x1 supported.");
		}
		
		int m = kxs.size();
		bool needFz = bool(z != 0.0);

		//Precalc Fx
		MatrixXd Fxprecalc(m, m);
		double *FxPrecalcPtr = Fxprecalc.data();

		for (int i = 0; i < m; i++) {
			Fxprecalc(i, i) = x1 - x0;
		}

		for (int i = 0; i < m; i++) {
			double kx = kxs(i);
			for (int j = i + 1; j < m; j++) {
				double kxP = kxs(j);
				double dk = kx - kxP;
				double Fx = CalcFxSym(dk, x0);
				FxPrecalcPtr[i * m + j] = Fx;
				FxPrecalcPtr[j * m + i] = Fx;
				//Fxprecalc(j, i) = Fx;
				//Fxprecalc(i, j) = Fx;
			}
		}

		// Precalc differentials
		ArrayXd dkxs(m);
		for (int i = 0; i < m; i++) {
			dkxs(i) = GetDifferential(kxs, i);
		}

		double integValue2dRe = 0.0;
		double integValue2dIm = 0.0;
		//#pragma omp parallel for default(shared) reduction(+:integValue2dRe,integValue2dIm)
		for (int i = 0; i < m; i++) {
			double kx = kxs(i);
			dcomplex kz = kzs(i);
			dcomplex U = Us(i);
			dcomplex integValue1d = 0.0;
			for (int j = 0; j < m; j++) {
				double kxP = kxs(j);
				dcomplex kzP = kzs(j);
				dcomplex UPc = std::conj(Us(j));
				double dkxP = dkxs(j);
				
				double Fx = FxPrecalcPtr[i * m + j];// Fxprecalc(j, i);
				dcomplex coef = dkxP * UPc * Fx;
				if (needFz) {
					coef *= FastExp(constI * (kz - kzP) * z);
				}

				// Integrate
				
				switch (pol)
				{
				case TMM::P_POL:
					integValue1d += coef * kz;
					break;
				case TMM::S_POL:
					integValue1d += coef * kzP;
					break;
				default:
					throw std::invalid_argument("Invalid polarization");
					break;
				}
			}
			double dkx = dkxs(i);
			dcomplex tmpRes = U * integValue1d * dkx;
			integValue2dRe += real(tmpRes);
			integValue2dIm += imag(tmpRes);
		}

		// Polarization specific multipliers
		dcomplex integValue2d(integValue2dRe, integValue2dIm);
		double omega = WlToOmega(wl);
		double res;

		switch (pol)
		{
		case TMM::P_POL:
			res = Ly / (2.0 * omega * constEps0) * std::real(integValue2d / epsLayer0);
			break;
		case TMM::S_POL:
			res = Ly / (2.0 * omega * constMu0) * real(integValue2d);
			break;
		default:
			throw std::invalid_argument("Invalid polarization");
			break;
		}

		return res;
	}
	*/

	pairdd const IntegrateWavePower(int layerNr, Polarization pol, double wl, dcomplex epsLayer, const Eigen::MatrixX2cd & Us, const ArrayXd & kxs, const Eigen::MatrixX2cd & kzs, double x0, double x1, double z, double Ly) {

		if (x0 != -x1 || z != 0) {
			std::cerr << "Currently only x0 = -x1, z = 0 supported." << std::endl;
			throw std::invalid_argument("Currently only x0 = -x1, z = 0 supported.");
		}

		int m = kxs.size();
		// Precalc differentials
		ArrayXd dkxs(m);
		for (int i = 0; i < m; i++) {
			dkxs(i) = GetDifferential(kxs, i);
		}

		// Not polarization specific
		dcomplex integValue2dF = 0.0;
		dcomplex integValue2dB = 0.0;
		for (int i = 0; i < m; i++) {
			dcomplex UF = Us(i, F);
			dcomplex kzF = kzs(i, F);
			dcomplex UB = Us(i, B);
			dcomplex kzB = kzs(i, B);
			double kx = kxs(i);
			double dkx = dkxs(i);			
			integValue2dF += dkx * dkx * UF * std::conj(UF) * kzF * (x1 - x0);
			integValue2dB += dkx * dkx * UB * std::conj(UB) * kzB * (x1 - x0);
		}

		// Polarization spetcific
		double omega = WlToOmega(wl);
		double resF, resB;
		if (pol == P_POL) {
			for (int i = 0; i < m; i++) {
				double kx = kxs(i);
				dcomplex kzF = kzs(i, F);
				dcomplex UF = Us(i, F);
				dcomplex kzB = kzs(i, B);
				dcomplex UB = Us(i, B);
				dcomplex integValue1dF = 0.0, integValue1dB = 0.0;
				for (int j = i + 1; j < m; j++) {
					double kxP = kxs(j);
					dcomplex kzPF = kzs(j, F);
					dcomplex UPF = Us(j, F);
					dcomplex kzPB = kzs(j, B);
					dcomplex UPB = Us(j, B);
					double dkxP = dkxs(j);
					double dk = kx - kxP;

					double cc = (std::sin(x0 * dk) * dkxP / dk);
					integValue1dF += cc * (UF * std::conj(UPF) * kzF + std::conj(UF) * UPF * kzPF);
					integValue1dB += cc * (UB * std::conj(UPB) * kzB + std::conj(UB) * UPB * kzPB);
				}
				double dkx = dkxs(i);
				integValue2dF -= (2.0 * dkx) * integValue1dF;
				integValue2dB -= (2.0 * dkx) * integValue1dB;
			}
			resF = Ly / (2.0 * omega * constEps0) * std::real(integValue2dF / epsLayer);
			resB = -Ly / (2.0 * omega * constEps0) * std::real(integValue2dB / epsLayer);
		} else if (pol == S_POL) {
			// Almost identical to p-polarization, copy for performance
			for (int i = 0; i < m; i++) {
				double kx = kxs(i);
				dcomplex kzF = kzs(i, F);
				dcomplex UF = Us(i, F);
				dcomplex kzB = kzs(i, B);
				dcomplex UB = Us(i, B);
				dcomplex integValue1dF = 0.0, integValue1dB = 0.0;
				for (int j = i + 1; j < m; j++) {
					double kxP = kxs(j);
					dcomplex kzPF = kzs(j, F);
					dcomplex UPF = Us(j, F);
					dcomplex kzPB = kzs(j, B);
					dcomplex UPB = Us(j, B);
					double dkxP = dkxs(j);
					double dk = kx - kxP;

					double cc = (std::sin(x0 * dk) * dkxP / dk);
					integValue1dF += cc * (UF * std::conj(UPF) * kzPF + std::conj(UF) * UPF * kzF);
					integValue1dB += cc * (UB * std::conj(UPB) * kzPB + std::conj(UB) * UPB * kzB);
				}
				double dkx = dkxs(i);
				integValue2dF -= (2.0 * dkx) * integValue1dF;
				integValue2dB -= (2.0 * dkx) * integValue1dB;

			}
			resF = Ly / (2.0 * omega * constMu0) * real(integValue2dF);
			resB = -Ly / (2.0 * omega * constMu0) * real(integValue2dB);
		} else {
			std::cerr << "Unknown polarization."  << std::endl;
			throw std::invalid_argument("Unknown polarization.");
		}

		return pairdd(resF, resB);
	}

	WaveDirection GetWaveDirection(dcomplex kzF, dcomplex eps, Polarization pol) {
		double pwrFlow;
		switch (pol)
		{
		case TMM::P_POL:
			pwrFlow = real(kzF / eps);
			break;
		case TMM::S_POL:
			pwrFlow = real(kzF);
			break;
		default:
			#pragma omp critical
			std::cerr << "Unknown polarization." << std::endl;
			throw std::invalid_argument("Unknown polarization.");
			break;
		}

		WaveDirection res;

		if (real(kzF) > 0.0) {
			res = F;
		}
		else if (real(kzF) < 0.0) {
			res = B;
		}
		else {
			if (imag(kzF) > 0.0) {
				res = F;
			}
			else if (imag(kzF) < 0.0) {
				res = B;
			}
			else {
				// Only if kz = 0.0
				#pragma omp critical
				std::cerr << "kzF = 0: " << kzF << " " << eps << ": " << pwrFlow << std::endl;
				throw std::runtime_error("Could not determine wave direction.");
			}
		}

		if (res == F && (/*imag(kzF) < 0.0 || */ pwrFlow < 0.0)) {
			#pragma omp critical
			std::cerr << "F: " << kzF << " " << eps << ": " << pwrFlow << std::endl;
			//throw std::runtime_error("Could not determine wave direction.");
		}

		if (res == B && (/*imag(kzF) > 0.0 || */pwrFlow > 0.0)) {
			#pragma omp critical
			std::cerr << "B: " << kzF << " " << eps << ": " << pwrFlow << std::endl;
			//throw std::runtime_error("Could not determine wave direction.");
		}
		return res;
	}

	ArrayXcd FFTShift(ArrayXcd data) {
		int p2 = (data.size() + 1) / 2;

		ArrayXcd res(data.size());
		for (int i = p2; i < data.size(); i++) {
			res(i - p2) = data(i);
		}
		for (int i = 0; i < p2; i++) {
			res(data.size() - p2 + i) = data(i);
		}
		return res;
	}

	ArrayXd IFFTShift(ArrayXd data) {
		int p2 = data.size() - (data.size() + 1) / 2;
		ArrayXd res(data.size());

		for (int i = p2; i < data.size(); i++) {
			res(i - p2) = data(i);
		}
		for (int i = 0; i < p2; i++) {
			res(data.size() - p2 + i) = data(i);
		}
		return res;
	}

	ArrayXd FFTFreq(int n, double dx) {
		ArrayXd res(n);
		int startI = -(n - (n % 2)) / 2;
		double step = 1.0 / (dx * n);
		for (int i = 0; i < n; i++) {
			res(i) = (startI + i) * step;
		}
		return res;
	}

};