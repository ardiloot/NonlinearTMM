#include "NonlinearLayer.h"

namespace TMM{

	Fields::Fields() {
		E = Vector3cd::Zero();
		H = Vector3cd::Zero();
		dir = TOT;
	}

	void Fields::SetFieldsPpol(const Array2cd &Hy, const Array2cd &Ex, const Array2cd &Ez, WaveDirection dir_)
	{
		E = Vector3cd::Zero();
		H = Vector3cd::Zero();
		dir = dir_;

		switch (dir)
		{
		case F:
			E(0) = Ex(F);
			E(2) = Ez(F);
			H(1) = Hy(F);
			break;
		case B:
			E(0) = Ex(B);
			E(2) = Ez(B);
			H(1) = Hy(B);
			break;
		case TOT:
			E(0) = Ex(F) + Ex(B);
			E(2) = Ez(F) + Ez(B);
			H(1) = Hy(F) + Hy(B);
			break;
		default:
			throw std::invalid_argument("Unknown wave direction.");
			break;
		}
	}
	void Fields::SetFieldsSpol(const Array2cd &Ey, const Array2cd &Hx, const Array2cd &Hz, WaveDirection dir_)
	{
		E = Vector3cd::Zero();
		H = Vector3cd::Zero();
		dir = dir_;

		switch (dir)
		{
		case F:
			E(1) = Ey(F);
			H(0) = Hx(F);
			H(2) = Hz(F);
			break;
		case B:
			E(1) = Ey(B);
			H(0) = Hx(B);
			H(2) = Hz(B);
			break;
		case TOT:
			E(1) = Ey(F) + Ey(B);
			H(0) = Hx(F) + Hx(B);
			H(2) = Hz(F) + Hz(B);
			break;
		default:
			throw std::invalid_argument("Unknown wave direction.");
			break;
		}
	}

	InhomogeneosWaveParams::InhomogeneosWaveParams() {
		kSzF = 0.0;
		pF = Vector3cd::Zero();
		pF = Vector3cd::Zero();
	}

	HomogeneousWave::HomogeneousWave(NonlinearLayer * layer_) {
		layer = layer_;
		kz = Array2cd::Zero();
		phase = Array2cd::Zero();
		propMatrix = Matrix2cd::Zero();
		solved = false;
	}

	void HomogeneousWave::Solve(NonlinearLayer * layer_) {
		layer = layer_;
		kz(F) = std::sqrt(dcomplex(layer->kNorm - sqr(layer->kx))); // Slowest part (AVX?)
		kz(B) = -kz(F);

		phase = Array2cd::Ones();
		if (!std::isinf(layer->d)) {
			phase = (constI * kz * layer->d).exp();
		}

		if (imag(kz(F)) < 0.0) {
			// Shold not hapen, checking
			std::cerr << "kzF imaginary part negative" << std::endl;
			throw std::runtime_error("kzF imaginary part negative");
		}

		if (real(kz(F)) < 0.0) {
			// Shold not hapen, checking
			std::cerr << "kzF real part negative" << std::endl;
			throw std::runtime_error("kzF real part negative");
		}

		// Propagation matrix
		if (isinf(layer->d)) {
			propMatrix = Matrix2cd::Identity();
		}
		else {
			propMatrix << phase(F), 0.0, 0.0, phase(B);
		}
		solved = true;
	}

	Array2cd HomogeneousWave::GetKz() const {
		if (!solved) {
			throw std::runtime_error("Homogeneous wave must be solved first");
		}
		return kz;
	}

	dcomplex HomogeneousWave::GetKzF() const{
		if (!solved) {
			throw std::runtime_error("Homogeneous wave must be solved first");
		}
		return  kz(F);
	}

	double HomogeneousWave::GetKx() const {
		if (!solved) {
			throw std::runtime_error("Homogeneous wave must be solved first");
		}
		return layer->kx;
	}

	Array2cd HomogeneousWave::GetMainFields(double z) const {
		if (!solved) {
			throw std::runtime_error("Homogeneous wave must be solved first");
		}
		Array2cd U = layer->U0;
		if (z == 0.0) {
			// No phase correction
		} else if (z == layer->d) {
			// Phase at the end of the layer is precalculated
			U *= phase;
		} else {
			U *= (constI * kz * z).exp();
		}
		return U;
	}

	InhomogeneousWave::InhomogeneousWave(NonlinearLayer * layer_) {
		layer = layer_;
		kSz = Array2cd::Zero();
		kSNorm = 0.0;
		phaseS = Array2cd::Zero();
		px = Array2cd::Zero();
		py = Array2cd::Zero();
		pz = Array2cd::Zero();
		By = Array2cd::Zero();
		propMatrixNL = Array2cd::Zero();
		solved = false;
	}

	void InhomogeneousWave::Solve(NonlinearLayer * layer_, InhomogeneosWaveParams & params) {
		layer = layer_;

		// kS
		kSz(F) = params.kSzF;
		kSz(B) = -kSz(F);
		kSNorm = sqr(layer->kx) + sqr(kSz(F));

		// Phase
		phaseS = Array2cd::Ones();
		if (!std::isinf(layer->d) && layer->IsNonlinear()) {
			phaseS = (constI * kSz * layer->d).exp();
		}

		// px, py, pz
		px << params.pF(0), params.pB(0);
		py << params.pF(1), params.pB(1);
		pz << params.pF(2), params.pB(2);

		// Calc By
		if (!layer->IsNonlinear()) {
			By << 0.0, 0.0;
		}
		else {
			if (layer->pol == S_POL) {
				By = -(py / constEps0) * sqr(layer->k0) / (layer->kNorm - kSNorm);
				By(B) *= phaseS(F);
			}
			else if (layer->pol == P_POL) {
				By = -layer->omega / (layer->kNorm - kSNorm) * (kSz * px - layer->kx * pz);
				By(B) *= phaseS(F);
			}
			else {
				throw std::runtime_error("Unknown polarization");
			}
		}

		// NL Propagation Matrix
		propMatrixNL = Array2cd::Zero();
		if (layer->IsNonlinear()) {
			if (isinf(layer->d)) {
				throw std::runtime_error("First and last medium must be nonlinear.");
			}
			else {
				propMatrixNL = By * (phaseS - layer->hw.phase);
			}
		}
		solved = true;
	}

	Array2cd InhomogeneousWave::GetMainFields(double z) const
	{
		if (!solved) {
			throw std::runtime_error("InhomogeneousWave wave must be solved first");
		}

		if (!layer->IsNonlinear()) {
			return Array2cd::Zero();
		}

		Array2cd U = By;
		if (z == 0.0) {
			// Phase is one
		} else if (z == layer->d) {
			// Phase at the end of the layer is precalculated
			U *= (phaseS - layer->hw.phase);
		} else {
			U *= ((constI * kSz * z).exp() - (constI * layer->hw.kz * z).exp());
		}
		return U;
	}

	NonlinearLayer::NonlinearLayer(double d_, Material *material_) {
		d = d_;
		material = material_;
		wl = 0.0;
		beta = 0.0;
		pol = NOT_DEFINED_POL;
		omega = 0.0;
		n = 0.0;
		eps = 0.0;
		k0 = 0.0;
		k = 0.0;
		kNorm = 0.0;
		kx = 0.0;
		propMatrix = Matrix2cd::Zero();
		propMatrixNL = Array2cd::Zero();
		isNonlinear = false;
		U0 = Array2cd::Zero();
		solved = false;
	}

	void NonlinearLayer::Solve(double wl_, double beta_, Polarization pol_) {
		wl = wl_;
		beta = beta_;
		pol = pol_;
		omega = WlToOmega(wl);

		// Find wavevectors
		n = material->GetN(wl);
		eps = sqr(n);
		k0 = 2.0 * PI / wl;
		k = k0 * n;
		kNorm = sqr(k);
		kx = beta * k0;

		// Solve waves
		hw.Solve(this);
		iws.Solve(this, kpS);
		iwa.Solve(this, kpA);

		// Propagation matrix
		propMatrix = hw.propMatrix;
		propMatrixNL = iwa.propMatrixNL + iws.propMatrixNL;
		solved = true;
	}

	bool NonlinearLayer::IsNonlinear() const
	{
		return isNonlinear;
	}

	void NonlinearLayer::SetNonlinearity(InhomogeneosWaveParams kpS_, InhomogeneosWaveParams kpA_)
	{
		isNonlinear = true;
		kpS = kpS_;
		kpA = kpA_;
	}

	void NonlinearLayer::ClearNonlinearity()
	{
		isNonlinear = false;
		kpS = InhomogeneosWaveParams();
		kpA = InhomogeneosWaveParams();
	}

	void NonlinearLayer::SetThickness(double d_) {
		d = d_;
	}

	void NonlinearLayer::SetParam(TMMParam param, double value) {
		switch (param)
		{
		case PARAM_LAYER_D:
			SetThickness(value);
			break;
		default:
			throw std::invalid_argument("Param not in list.");
			break;
		}
	}

	double NonlinearLayer::GetThickness() {
		return d;
	}

	Material * NonlinearLayer::GetMaterial() {
		// It is the responsibility of the user to ensure, that material ptr stays valid
		return material;
	}

	HomogeneousWave *NonlinearLayer::GetHw()
	{
		return &hw;
	}

	double NonlinearLayer::GetKx() const {
		if (!solved) {
			throw std::runtime_error("NonlinearLayer must be solved first.");
		}
		return kx;
	}

	double NonlinearLayer::GetK0() const {
		if (!solved) {
			throw std::runtime_error("NonlinearLayer must be solved first.");
		}
		return k0;
	}

	double NonlinearLayer::GetDouble(TMMParam param) {
		switch (param)
		{
		case PARAM_LAYER_D:
			return GetThickness();
			break;
		default:
			throw std::invalid_argument("Param not in list.");
			break;
		}
	}

	Array2cd NonlinearLayer::GetMainFields(double z) const
	{
		if (!solved) {
			throw std::runtime_error("NonlinearLayer must be solved first.");
		}

		Array2cd U = hw.GetMainFields(z) + iws.GetMainFields(z) + iwa.GetMainFields(z);
		return U;
	}

	Fields NonlinearLayer::GetFields(double z, WaveDirection dir) const
	{
		if (!solved) {
			throw std::runtime_error("NonlinearLayer must be solved first.");
		}

		Fields res;

		Array2cd expKSz = Array2cd::Ones();
		Array2cd expKAz = Array2cd::Ones();

		if (z == 0.0) {
			// Phase is 1
		}
		else if (z == d) {
			// End of layer phases are precalculated
			expKSz = iws.phaseS;
			expKAz = iwa.phaseS;
		} else {
			expKSz = (constI * iws.kSz * z).exp();
			expKAz = (constI * iwa.kSz * z).exp();
		}

		if (pol == P_POL) {
			dcomplex c = -1.0 / (omega * eps * constEps0);
			Array2cd Hy = GetMainFields(z);
			Array2cd Ex = (-hw.kz * Hy) * c;
			Array2cd Ez = (kx * Hy) * c;
			if (IsNonlinear()) {
				Ex += (iws.By * (hw.kz - iws.kSz) * expKSz + iws.px * expKSz * omega +
					iwa.By * (hw.kz - iwa.kSz) * expKAz + iwa.px * expKAz * omega) * c;
				Ez += (iws.pz * expKSz + iwa.pz * expKAz) * c * omega;
			}
			res.SetFieldsPpol(Hy, Ex, Ez, dir);
		}
		else if (pol == S_POL) {
			double c = 1.0 / (omega * constMu0);
			Array2cd Ey = GetMainFields(z);
			Array2cd Hx = (-hw.kz * Ey) * c;
			Array2cd Hz = (kx * Ey) * c;
			if (IsNonlinear()) {
				Hx += (iws.By * (hw.kz - iws.kSz) * expKSz +
					iwa.By * (hw.kz - iwa.kSz) * expKAz) * c;
			}
			res.SetFieldsSpol(Ey, Hx, Hz, dir);
		}
		else {
			throw std::runtime_error("Unknown polarization.");
		}
		return res;
	}

	double NonlinearLayer::GetIntensity(double z) const{
		if (!solved) {
			throw std::runtime_error("NonlinearLayer must be solved first.");
		}
		Fields f = GetFields(z, TOT);
		double Sz = 0.5 * real(f.E(0) * std::conj(f.H(1)) - f.E(1) * std::conj(f.H(0)));
		return Sz;
	}

	double NonlinearLayer::GetAbsorbedIntensity() const {
		if (!solved) {
			throw std::runtime_error("NonlinearLayer must be solved first.");
			std::cerr << "NonlinearLayer must be solved first." << std::endl;
		}

		if (abs(imag(eps)) < 1e-14) {
			//Layer not absorbing
			return 0.0;
		}

		if (IsNonlinear()) {
			throw std::runtime_error("Absorbed power calculation is not allowed in nonlinear media.");
			std::cerr << "Absorbed power calculation is not allowed in nonlinear media." << std::endl;
		}

		dcomplex kzF = hw.kz(F);
		dcomplex dk1 = kzF - std::conj(kzF);
		dcomplex dk2 = kzF + std::conj(kzF);
		dcomplex intValue1 = -constI * (std::exp(constI * dk1 * d) - 1.0) / dk1;
		dcomplex intValue2 = -constI * (std::exp(constI * dk2 * d) - 1.0) / dk2;
		dcomplex intValue3 = constI * (std::exp(-constI * dk2 * d) - 1.0) / dk2;
		dcomplex intValue4 = constI * (std::exp(-constI * dk1 * d) - 1.0) / dk1;

		Vector3cd E0F = GetFields(0.0, F).E;
		Vector3cd E0B = GetFields(0.0, B).E;
		Vector3cd E0FC = E0F.conjugate();
		Vector3cd E0BC = E0B.conjugate();

		// E0F.dot(E0FC) provided wrong results
		dcomplex dot1 = E0F(0) * E0FC(0) + E0F(1) * E0FC(1) + E0F(2) * E0FC(2);
		dcomplex dot2 = E0F(0) * E0BC(0) + E0F(1) * E0BC(1) + E0F(2) * E0BC(2);
		dcomplex dot3 = E0B(0) * E0FC(0) + E0B(1) * E0FC(1) + E0B(2) * E0FC(2);
		dcomplex dot4 = E0B(0) * E0BC(0) + E0B(1) * E0BC(1) + E0B(2) * E0BC(2);
		dcomplex intValue = dot1 * intValue1 + dot2 * intValue2 + dot3 * intValue3 + dot4 * intValue4;
		double res = 0.5 * constEps0 * imag(eps) * omega * real(intValue);

		return res;
	}

	double NonlinearLayer::GetSrcIntensity() const{
		if (!solved) {
			throw std::runtime_error("NonlinearTMM must be solved first.");
		}

		double absorbed = GetAbsorbedIntensity();
		double deltaS = 0.0;
		if (!std::isinf(d)) {
			deltaS = GetIntensity(d) - GetIntensity(0.0);
		}

		double res = absorbed + deltaS;
		return res;
	}

	double NonlinearLayer::GetENorm(double z, WaveDirection dir) const {
		Fields f = GetFields(z, dir);
		double res = std::sqrt(norm(f.E(0)) + norm(f.E(1)) + norm(f.E(2)));//f.E.norm();
		return res;
	}
	
};