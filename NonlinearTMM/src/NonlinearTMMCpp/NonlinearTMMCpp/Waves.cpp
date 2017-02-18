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

	void Wave::SolvePlaneWave(double E0, double th0, double k) {
		// Plane waves
		kxs = ArrayXd(1);
		kzs = ArrayXd(1);
		expansionCoefsKx = ArrayXcd(1);
		fieldProfileXs = ArrayXd(0);
		fieldProfile = ArrayXd(0);
		kxs(0) = k * std::sin(th0);
		kzs(0) = k * std::cos(th0);
		expansionCoefsKx(0) = E0;
	}

	void Wave::SolveFFTWave(double E0, double th0, double k, int iteration, double maxPhiThis) {
		// FFT waves
		double maxKxpByMaxX = (nPointsInteg - 1) / 2 * (PI / maxX);
		double maxKxp = min(maxKxpByMaxX, std::sin(maxPhiThis) * k);
		double dx = PI / maxKxp;

		//std::cout << "x: " << dx * (nPointsInteg - 1) << std::endl;

		// Init xs
		ArrayXd xs(nPointsInteg);
		for (int i = 0; i < nPointsInteg; i++) {
			xs(i) = -0.5 * dx * nPointsInteg + i * dx;
		}

		// Field profile
		fieldProfileXs = xs;
		switch (waveType)
		{
		case TMM::GAUSSIANWAVE:
			fieldProfile = E0 * (-(xs * xs) / sqr(w0)).exp();
			break;
		case TMM::TUKEYWAVE:
			fieldProfile = E0 * TukeyFunc(xs, w0, a);
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
		ArrayXd phis = (kxPs / k).asin();


		// Check if needed to reduce maxPhi
		ArrayXd cumSum(fieldProfileSpectrum.size());
		cumSum(0) = std::abs(fieldProfileSpectrum(0));
		for (int i = 1; i < cumSum.size(); i++) {
			cumSum(i) = cumSum(i - 1) + std::abs(fieldProfileSpectrum(i));
		}

		double newMaxPhiIndex = -1;
		double totalSum = cumSum(cumSum.size() - 1);
		for (int i = 1; i < cumSum.size(); i++) {
			if (cumSum(i) / totalSum >= integCriteria) {
				newMaxPhiIndex = i;
				break;
			}
		}

		// Recursion if can reduce maxPhi
		if (newMaxPhiIndex > 0 && iteration < 1) {
			double newMaxPhi = abs(phis(max(0, newMaxPhiIndex - 5)));
			SolveFFTWave(E0, th0, k, iteration + 1, newMaxPhi);
			return;
		}

		// Check for backward-propagating waves
		if (phis.maxCoeff() + th0 >= PI / 2.0) {
			std::cerr << "Phi larger than 90 deg." << std::endl;
			throw std::runtime_error("Phi larger than 90 deg.");
		}

		// Save result
		ArrayXd kzPs = phis.cos() * k;
		kxs = kxPs * std::cos(th0) + kzPs * std::sin(th0);
		kzs = -kxPs * std::sin(th0) + kzPs * std::cos(th0);
		expansionCoefsKx = fieldProfileSpectrum / std::cos(th0);
	}

	Wave::Wave() : kxs(0), kzs(0), expansionCoefsKx(0), fieldProfileXs(0), fieldProfile(0) {
		waveType = PLANEWAVE;
		pwr = 1.0;
		overrideE0 = false;
		E0OverrideValue = 1.0;
		w0 = 100e-6;
		material = NULL;
		Ly = 1e-3;
		a = 0.7;
		nPointsInteg = 100;
		maxPhi = 0.17;
		integCriteria = 1e-3;
		maxX = 1e-3;
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

	void Wave::SetMaterial(Material * material_) {
		material = material_;
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

	void Wave::SetMaxPhi(double maxPhi_) {
		maxPhi = maxPhi_;
	}

	void Wave::SetIntegCriteria(double criteria_) {
		integCriteria = criteria_;
	}

	void Wave::SetMaxX(double maxX_) {
		maxX = maxX_;
	}

	void Wave::Solve(double wl_, double beta_, Material *material_) {
		wl = wl_;
		beta = beta_;

		if (material_ != NULL) {
			SetMaterial(material_);
		}
		
		if (material == NULL) {
			std::cerr << "Material is not set." << std::endl;
			throw std::runtime_error("Material is not set.");
		}

		// E0
		double E0;
		if (!overrideE0) {
			double I0 = pwr / (Ly * w0);
			E0 = std::sqrt(2.0 * constMu0 * constC * I0);
		}
		else {
			E0 = E0OverrideValue;
		}

		k0 = 2.0 * PI / wl;
		double n = real(material->GetN(wl));
		double k = k0 * n;
		double th0 = std::asin(beta / n);

		if (waveType == PLANEWAVE) {
			SolvePlaneWave(E0, th0, k);
		}
		else {
			SolveFFTWave(E0, th0, k, 0, maxPhi);
		}
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

	double Wave::GetMaxPhi() {
		return maxPhi;
	}

	double Wave::GetIntegCriteria() {
		return integCriteria;
	}

	double Wave::GetMaxX() {
		return maxX;
	}

	ArrayXd Wave::GetBetas() {
		return kxs / k0;
	}

	ArrayXd Wave::GetKxs() {
		return kxs;
	}

	ArrayXd Wave::GetKzs() {
		return kzs;
	}

	ArrayXd Wave::GetFieldProfileXs() {
		return fieldProfileXs;
	}

	ArrayXd Wave::GetFieldProfile() {
		return fieldProfile;
	}

	ArrayXcd Wave::GetExpansionCoefsKx() {
		return expansionCoefsKx;
	}

}