#include "Waves.h"

namespace TMM {

	ArrayXd TukeyFunc(ArrayXd xs, double w0, double a) {
		ArrayXd res(xs.size());
		for (int i = 0; i < xs.size(); i++) {
			double x = xs(i);

			if (x < -0.5 * w0 * (2.0 - a)) {
				res(i) = 0.0;
			}
			else if (x > 0.5 * w0 * (2.0 - a)) {
				res(i) = 0.0;
			} else if (x > 0.5 * a * w0) {
				res(i) = 0.5 * std::cos(PI / (w0 * (1.0 - a)) * (x - 0.5 * a * w0)) + 0.5;
			}
			else if (x < -0.5 * a * w0) {
				res(i) = 0.5 * std::cos(PI / (w0 * (1.0 - a)) * (x + 0.5 * (2.0 - a) * w0) + PI) + 0.5;
			}
			else {
				res(i) = 1.0;
			}
		}
		return res;
	}

	void Wave::SolvePlaneWave() {
		maxXThis = (0.5 * w0 / std::cos(thLayer0));

		// Plane waves
		phis = ArrayXd(1);
		kxs = ArrayXd(1);
		kzs = ArrayXd(1);
		expansionCoefsKx = ArrayXcd(1);
		fieldProfileXs = ArrayXd(0);
		fieldProfile = ArrayXd(0);
		phis(0) = 0.0;
		kxs(0) = k * std::sin(thLayer0);
		kzs(0) = k * std::cos(thLayer0);
		expansionCoefsKx(0) = E0;
	}

	void Wave::SolveFFTWave() {
		if (nPointsInteg < 2) {
			std::cerr << "nPointsInteg too small" << std::endl;
			throw std::runtime_error("nPointsInteg too small");
		}
		
		double maxXByPhi = (nPointsInteg / 2) * PI / (std::sin(maxPhi) * k);
		// Calc maxX
		if (dynamicMaxX) {
			maxXThis = dynamicMaxXCoef * (0.5 * w0 / std::cos(thLayer0)) + dynamicMaxXAddition;
		}
		else {
			maxXThis = maxX;
		}
		maxXThis = max(maxXThis, maxXByPhi);
		
		// Init xs
		ArrayXd xs(nPointsInteg);
		xs = ArrayXd::LinSpaced(nPointsInteg, -maxXThis, maxXThis);
		double dx = xs(1) - xs(0);

		// Field profile
		fieldProfileXs = xs;
		switch (waveType)
		{
		case TMM::GAUSSIANWAVE:
			fieldProfile = E0 * (-(xs * xs) / sqr(w0 / 1.2533141369277430)).exp();
			break;
		case TMM::TUKEYWAVE:
			fieldProfile = E0 * TukeyFunc(xs, w0 / (0.25 * a + 0.75), a);
			break;
		default:
			std::cerr << "Unknown FFT wave type." << std::endl;
			throw std::invalid_argument("Unknown FFT wave type.");
			break;
		}

		// FFT spectrum
		ArrayXcd fieldProfileSpectrum(fieldProfile.size());
		fft.fwd(fieldProfileSpectrum.matrix(), IFFTShift(fieldProfile).matrix());
		fieldProfileSpectrum = FFTShift(fieldProfileSpectrum);
		fieldProfileSpectrum *= dx / (2.0 * PI);
		ArrayXd kxPs = 2.0 * PI * FFTFreq(fieldProfile.size(), dx);
		phis = (kxPs / k).asin();

		// Check for backward-propagating waves
		if (phis.maxCoeff() + thLayer0 >= PI / 2.0) {
			std::cerr << "Phi larger than 90 deg." << std::endl;
			throw std::runtime_error("Phi larger than 90 deg.");
		}

		// Save result
		ArrayXd kzPs = phis.cos() * k;
		kxs = kxPs * std::cos(thLayer0) + kzPs * std::sin(thLayer0);
		kzs = -kxPs * std::sin(thLayer0) + kzPs * std::cos(thLayer0);
		expansionCoefsKx = fieldProfileSpectrum / std::cos(thLayer0);
	}

	void Wave::SolveSpdcWave() {
		// Checks
		if (std::isnan(deltaKxSpdc)) {
			std::cerr << "SPDC wave can only be used through SecondOrderNLTMM" << std::endl;
			throw std::runtime_error("SPDC wave can only be used through SecondOrderNLTMM");
		}
		maxXThis = (0.5 * w0 / std::cos(thLayer0));

		// Init arrays
		phis = ArrayXd(1);
		fieldProfileXs = ArrayXd(0);
		fieldProfile = ArrayXd(0);

		double kxMid = k * std::sin(thLayer0);
		kxs = ArrayXd::LinSpaced(nPointsInteg, kxMid - deltaKxSpdc, kxMid + deltaKxSpdc);
		kzs = (k * k - kxs.pow(2)).sqrt();
		expansionCoefsKx = ArrayXcd::Ones(nPointsInteg);
		phis(0) = 0.0; // Currently not calculated (not needed)
	}

	Wave::Wave() : phis(0), kxs(0), kzs(0), expansionCoefsKx(0), fieldProfileXs(0), fieldProfile(0) {
		waveType = PLANEWAVE;
		pwr = 1.0;
		overrideE0 = false;
		E0OverrideValue = 1.0;
		w0 = 100e-6;
		materialLayer0 = NULL;
		materialLayerThis = NULL;
		Ly = 1e-3;
		a = 0.7;
		nPointsInteg = 100;
		maxX = 1e-3;
		dynamicMaxX = true;
		dynamicMaxXCoef = 2.0;
		dynamicMaxXAddition = 0.0;
		maxXThis = 0.0;
		solved = false;
		maxPhi = 0.17;
		deltaKxSpdc = constNAN;

		wl = 0.0;
		beta = 0.0;
		k0 = 0.0;
		E0 = 0.0;
		k = 0.0;
		nLayer0 = 0.0;
		thLayer0 = 0.0;
	}

	void Wave::SetWaveType(WaveType waveType_) {
		waveType = waveType_;
	}

	void Wave::SetPwr(double pwr_) {
		pwr = pwr_;
	}

	void Wave::SetOverrideE0(bool overrideE0_) {
		overrideE0 = overrideE0_;
	}

	void Wave::SetE0(double E0_) {
		E0OverrideValue = E0_;
	}

	void Wave::SetW0(double w0_) {
		w0 = w0_;
	}

	void Wave::SetLy(double Ly_) {
		Ly = Ly_;
	}

	void Wave::SetA(double a_) {
		a = a_;
	}

	void Wave::SetNPointsInteg(int nPointsInteg_) {
		nPointsInteg = nPointsInteg_;
	}

	void Wave::SetMaxX(double maxX_) {
		maxX = maxX_;
	}

	void Wave::EnableDynamicMaxX(bool dynamicMaxX_) {
		dynamicMaxX = dynamicMaxX_;
	}

	void Wave::SetDynamicMaxXCoef(double dynamicMaxXCoef_) {
		dynamicMaxXCoef = dynamicMaxXCoef_;
	}

	void Wave::SetDynamicMaxXAddition(double dynamicMaxXAddition_) {
		dynamicMaxXAddition = dynamicMaxXAddition_;
	}

	void Wave::SetMaxPhi(double maxPhi_) {
		maxPhi = maxPhi_;
	}

	void Wave::SetParam(TMMParam param, double value) {
		switch (param)
		{
		case PARAM_WAVE_W0:
			SetW0(value);
			break;
		default:
			throw std::invalid_argument("Param not in list.");
			break;
		}
	}

	void Wave::Solve(double wl_, double beta_, Material *materialLayer0_, Material *materialLayerThis_, double deltaKxSpdc_) {
		wl = wl_;
		beta = beta_;
		materialLayer0 = materialLayer0_;
		materialLayerThis = materialLayerThis_;
		deltaKxSpdc = deltaKxSpdc_;
		maxXThis = maxX;
		
		// Calc E0
		if (!overrideE0) {
			double I0 = pwr / (Ly * w0);
			E0 = std::sqrt(2.0 * constMu0 * constC * I0);
		}
		else {
			E0 = E0OverrideValue;
		}

		// Precalc variables
		k0 = 2.0 * PI / wl;
		nLayer0 = real(materialLayer0->GetN(wl));
		k = k0 * nLayer0;
		thLayer0 = std::asin(beta / nLayer0);
		beamArea = w0 * Ly / std::cos(thLayer0);

		// Solve
		if (waveType == PLANEWAVE) {
			SolvePlaneWave();
		}
		else if (waveType == GAUSSIANWAVE || waveType == TUKEYWAVE) {
			SolveFFTWave();
		}
		else if (waveType == SPDCWAVE) {
			SolveSpdcWave();
		}
		else {
			std::cerr << "Unknown wave type." << std::endl;
			throw std::invalid_argument("Unknown wave type.");
		}
		solved = true;
	}


	// Getters

	WaveType Wave::GetWaveType() {
		return waveType;
	}

	double Wave::GetPwr() {
		return pwr;
	}

	bool Wave::GetOverrideE0() {
		return overrideE0;
	}

	double Wave::GetE0() {
		return E0OverrideValue;
	}

	double Wave::GetW0() {
		return w0;
	}

	double Wave::GetLy() {
		return Ly;
	}

	double Wave::GetA() {
		return a;
	}

	int Wave::GetNPointsInteg() {
		return nPointsInteg;
	}

	double Wave::GetMaxX() {
		return maxX;
	}

	bool Wave::IsDynamicMaxXEnabled() {
		return dynamicMaxX;
	}

	double Wave::GetDynamicMaxXCoef() {
		return dynamicMaxXCoef;
	}

	double Wave::GetDynamicMaxXAddition() {
		return dynamicMaxXAddition;
	}

	double Wave::GetMaxPhi() {
		return maxPhi;
	}

	double Wave::GetDouble(TMMParam param) {
		switch (param)
		{
		case PARAM_WAVE_W0:
			return GetW0();
			break;
		default:
			throw std::invalid_argument("Param not in list.");
			break;
		}
	}

	pairdd Wave::GetXRange() {
		if (!solved) {
			std::cerr << "Wave must be solved first." << std::endl;
			throw std::runtime_error("Wave must be solved first.");
		}
		return pairdd(-maxXThis, maxXThis);
	}

	ArrayXd Wave::GetBetas() {
		if (!solved) {
			std::cerr << "Wave must be solved first." << std::endl;
			throw std::runtime_error("Wave must be solved first.");
		}
		return kxs / k0;
	}

	ArrayXd Wave::GetPhis() { return phis; }

	ArrayXd Wave::GetKxs() {
		if (!solved) {
			std::cerr << "Wave must be solved first." << std::endl;
			throw std::runtime_error("Wave must be solved first.");
		}
		return kxs;
	}

	ArrayXd Wave::GetKzs() {
		if (!solved) {
			std::cerr << "Wave must be solved first." << std::endl;
			throw std::runtime_error("Wave must be solved first.");
		}
		return kzs;
	}

	ArrayXd Wave::GetFieldProfileXs() {
		if (!solved) {
			std::cerr << "Wave must be solved first." << std::endl;
			throw std::runtime_error("Wave must be solved first.");
		}
		return fieldProfileXs;
	}

	ArrayXd Wave::GetFieldProfile() {
		if (!solved) {
			std::cerr << "Wave must be solved first." << std::endl;
			throw std::runtime_error("Wave must be solved first.");
		}
		return fieldProfile;
	}

	ArrayXcd Wave::GetExpansionCoefsKx() {
		if (!solved) {
			std::cerr << "Wave must be solved first." << std::endl;
			throw std::runtime_error("Wave must be solved first.");
		}
		return expansionCoefsKx;
	}

	double Wave::GetBeamArea() {
		if (!solved) {
			std::cerr << "Wave must be solved first." << std::endl;
			throw std::runtime_error("Wave must be solved first.");
		}
		return beamArea;
	}

	bool Wave::IsCoherent() {
		if (waveType == SPDCWAVE) {
			return false;
		}
		return true;
	}

}