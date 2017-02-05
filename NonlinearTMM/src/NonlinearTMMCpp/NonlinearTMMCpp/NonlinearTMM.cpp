#include "NonlinearTmm.h"

namespace TMM {

	void OuterProductSSEEigenComplex(const Eigen::ArrayXcd & X, const Eigen::ArrayXcd & Y, Eigen::MatrixXcd & R) {
		register dcomplex* ptrR = &R(0, 0);
		const register dcomplex* ptrX = (dcomplex*)&X(0);
		const register dcomplex* ptrY = (dcomplex*)&Y(0);
		const register int n = X.size();
		const register int m = Y.size();

		for (int i = 0; i < n; i++) {
			register int k = i * m;
			register dcomplex v = ptrX[i];
			for (int j = 0; j < m; j++) {
				register int k2 = k + j;
				register dcomplex w = ptrY[j];
				ptrR[k2] = multSSE(v, w);
			}
		}
	}

	void OuterProductGoodEigenComplex(const Eigen::ArrayXcd & X, const Eigen::ArrayXcd & Y, Eigen::MatrixXcd & R) {
		dcomplex* ptrR = &R(0, 0);
		dcomplex* ptrX = (dcomplex*)&X(0);
		dcomplex* ptrY = (dcomplex*)&Y(0);
		int n = X.size();
		int m = Y.size();

		for (int i = 0; i < n; i++) {
			int k = i * m;
			dcomplex v = ptrX[i];
			for (int j = 0; j < m; j++) {
				dcomplex w = ptrY[j];
				int k2 = k + j;
				ptrR[k2] = v * w;
			}
		}
	}

	PowerFlows::PowerFlows(dcomplex inc_, dcomplex r_, dcomplex t_, double I_, double R_, double T_) {
		inc = inc_;
		r = r_;
		t = t_;
		I = I_;
		R = R_;
		T = T_;
	}

	std::ostream & operator<<(std::ostream & os, const PowerFlows &p)
	{
		os << p.inc << " " << p.r << " " << p.t << " : " << p.I << " " << p.R << " " << p.T;
		return os;
	}

	SweepResultNonlinearTMM::SweepResultNonlinearTMM(int n) : inc(n), r(n), t(n), I(n), R(n), T(n) {
	}

	void SweepResultNonlinearTMM::SetPowerFlows(int nr, const PowerFlows& pw) {
		inc(nr) = pw.inc;
		r(nr) = pw.r;
		t(nr) = pw.t;
		I(nr) = pw.I;
		R(nr) = pw.R;
		T(nr) = pw.T;
	}

	FieldsZ::FieldsZ(int n) : E(n, 3), H(n, 3) {
		// Not zereod because of performance
	}

	void FieldsZ::SetFieldsAtZ(int nr, const Fields & f) {
		E.row(nr) = f.E;
		H.row(nr) = f.H;
	}

	FieldsZX::FieldsZX(int n, int m, Polarization pol_) : Ex(n, m), Ey(n, m), Ez(n, m), Hx(n, m), Hy(n, m), Hz(n, m) {
		pol = pol_;
		// Fields not zereod because of performance
	}

	Polarization FieldsZX::GetPol() {
		return pol;
	}

	void FieldsZX::SetFields(const FieldsZ & F, const Eigen::ArrayXcd & phaseX, Polarization pol) {
		int n = F.E.rows();
		int m = phaseX.size();

		// Parallelisation has no/small effect
		switch (pol)
		{
		case TMM::P_POL:
			OuterProductSSEEigenComplex(phaseX, F.E.col(0), Ex);
			OuterProductSSEEigenComplex(phaseX, F.E.col(2), Ez);
			OuterProductSSEEigenComplex(phaseX, F.H.col(1), Hy);
			// S-pol fields stay undefined because of performance
			break;
		case TMM::S_POL:
			OuterProductSSEEigenComplex(phaseX, F.E.col(1), Ey);
			OuterProductSSEEigenComplex(phaseX, F.H.col(0), Hx);
			OuterProductSSEEigenComplex(phaseX, F.H.col(2), Hz);
			// P-pol fields stay undefined because of performance
			break;
		default:
			throw std::invalid_argument("Unknown polarization.");
			break;
		}
	}

	void NonlinearTMM::CheckPrerequisites(TMMParam toIgnore) {
		// Check params
		if (toIgnore != PARAM_WL && isnan(wl)) {
			throw std::invalid_argument("Wavelength is not set.");
		}

		if (toIgnore != PARAM_BETA && isnan(beta)) {
			throw std::invalid_argument("Beta is not set.");
		}

		if (toIgnore != PARAM_POL && pol == NOT_DEFINED_POL) {
			throw std::invalid_argument("Polarization is not set.");
		}

		if (layers.size() < 2) {
			throw std::invalid_argument("TMM must have at least 2 layers.");
		}
	}

	Array2cd NonlinearTMM::CalcTransferMatrixNL(int interfaceNr, const InhomogeneousWave & w1, const InhomogeneousWave & w2) {
		NonlinearLayer &l1 = layers[interfaceNr];
		NonlinearLayer &l2 = layers[interfaceNr + 1];

		if (!l1.IsNonlinear() && !l2.IsNonlinear()) {
			// Both media are linear
			return Array2cd::Zero();
		}

		dcomplex f1 = (w1.kSz(F) - l1.hw.kz(F)) * (w1.By(F) * w1.phaseS(0) - w1.By(B) * w1.phaseS(1));
		dcomplex f2 = (w2.kSz(F) - l2.hw.kz(F)) * (w2.By(B) - w2.By(F));
		dcomplex f3 = 0.0, cNL;
		if (pol == P_POL) {
			f3 = -l1.omega * l2.eps * (w1.px(B) * w1.phaseS(1) + w1.px(F) * w1.phaseS(0)) + l1.omega * l1.eps * (w2.px(B) + w2.px(F));
			cNL = 0.5 / (l1.eps * l2.hw.kz(F)) * (l2.eps * f1 + l1.eps * f2 + f3);
		}
		else if (pol == S_POL) {
			cNL = 0.5 * (f1 + f2) / (l2.hw.kz(F));
		}
		else {
			throw std::runtime_error("Unknown polarization.");
		}

		Array2cd transferMatrixNL;
		transferMatrixNL << cNL, -cNL;
		return transferMatrixNL;
	}

	void NonlinearTMM::SolveInterfaceTransferMatrix(int interfaceNr) {

		NonlinearLayer &l1 = layers[interfaceNr];
		NonlinearLayer &l2 = layers[interfaceNr + 1];

		// Polarization specific
		dcomplex a;
		if (pol == P_POL) {
			a = (l2.eps * l1.hw.kz(F)) / (l1.eps * l2.hw.kz(F));
		}
		else if (pol == S_POL) {
			a = l1.hw.kz(F) / l2.hw.kz(F);
		}
		else {
			throw std::runtime_error("Unknown polarization");
		}

		//Transfer matrices
		transferMatrices[interfaceNr] << 0.5 + 0.5 * a, 0.5 - 0.5 * a, 0.5 - 0.5 * a, 0.5 + 0.5 * a;
		transferMatricesNL[interfaceNr] = CalcTransferMatrixNL(interfaceNr, l1.iws, l2.iws) + CalcTransferMatrixNL(interfaceNr, l1.iwa, l2.iwa);
	}

	void NonlinearTMM::SolveAllTransferMatrices() {
		// Calc all
		for (int i = 0; i < layers.size() - 1; i++) {
			SolveInterfaceTransferMatrix(i);
		}
	}

	void NonlinearTMM::SolveSystemMatrix() {
		// Calc
		systemMatrices[0] = Matrix2cd::Identity();
		systemMatricesNL[0] = Array2cd::Zero();
		for (int i = 0; i < layers.size() - 1; i++) {
			NonlinearLayer &layer = layers[i];
			Matrix2cd &transferL = transferMatrices[i];
			Array2cd &transferNL = transferMatricesNL[i];

			systemMatrices[i + 1] = transferL * layer.propMatrix * systemMatrices[i];
			Array2cd mSysNLProp = (layer.propMatrix * systemMatricesNL[i].matrix()).array() + layer.propMatrixNL;
			systemMatricesNL[i + 1] = (transferL * mSysNLProp.matrix()).array() + transferNL;
		}
	}

	void NonlinearTMM::SolveIncReflTrans() {
		// System matrix
		Matrix2cd &mSysL = systemMatrices[layers.size() - 1];
		Array2cd &mSysNL = systemMatricesNL[layers.size() - 1];

		// Calc reflection and transmission
		double omega = WlToOmega(wl);
		NonlinearLayer &l0 = layers[0];
		NonlinearLayer &lL = layers[layers.size() - 1];
		if (mode == MODE_INCIDENT) {
			if (!overrideE0) {
				// Intensity given by I0
				if (pol == P_POL) {
					inc = std::sqrt(2.0 * omega * constEps0 / std::real(l0.hw.kz(F)) * real(l0.eps) * I0);
				}
				else if (pol == S_POL) {
					inc = std::sqrt(2.0 * omega * constMu0 / std::real(l0.hw.kz(F)) * I0);
				}
				else {
					throw std::runtime_error("Unknown polarization");
				}
			}
			else {
				// Pump wave E0 is given in vacuum
				if (pol == P_POL) {
					inc = E0 * std::sqrt(real(l0.n)) * constEps0 * constC;
				}
				else if (pol == S_POL) {
					inc = E0 / std::sqrt(real(l0.n));
				}
				else {
					throw std::runtime_error("Unknown polarization");
				}
			}
			r = -inc * mSysL(1, 0) / mSysL(1, 1);
			t = inc * mSysL.determinant() / mSysL(1, 1);
		}
		else if (mode == MODE_NONLINEAR) {
			// Nonlinear mode
			inc = 0.0;
			r = -mSysNL(1, 0) / mSysL(1, 1);
			t = mSysNL(0, 0) - mSysL(0, 1) / mSysL(1, 1) * mSysNL(1, 0);
		}
		else {
			throw std::runtime_error("Unknown mode.");
		}
	}

	NonlinearTMM::NonlinearTMM() {
		wl = constNAN;
		beta = constNAN;
		pol = NOT_DEFINED_POL;
		I0 = 1.0;
		overrideE0 = false;
		E0 = 1.0;
		mode = MODE_INCIDENT;
		layers.reserve(7);
		systemMatrices.reserve(7);
		systemMatricesNL.reserve(7);
		transferMatrices.reserve(7);
		transferMatricesNL.reserve(7);
		inc = 0.0;
		r = 0.0;
		t = 0.0;
		solved = false;
	}

	void NonlinearTMM::AddLayer(double d_, Material *material_) {
		if (layers.size() == 0 && !isinf(d_)) {
			throw std::invalid_argument("First layer must have infinite thickness.");
		}

		layers.push_back(NonlinearLayer(d_, material_));
		systemMatrices.resize(layers.size());
		systemMatricesNL.resize(layers.size());
		transferMatrices.resize(layers.size());
		transferMatricesNL.resize(layers.size());
	}


	NonlinearLayer* NonlinearTMM::GetLayer(int layerNr)
	{
		if (layerNr < 0 || layerNr >= layers.size()) {
			throw std::invalid_argument("Layer index invalid.");
		}
		return &layers[layerNr];
	}

	int NonlinearTMM::LayersCount() const {
		return layers.size();
	}

	void NonlinearTMM::Solve() {
		CheckPrerequisites();

		// Solve all layers
		for (int i = 0; i < layers.size(); i++) {				
			layers[i].Solve(wl, beta, pol);
		}

		// Solve
		SolveAllTransferMatrices();
		SolveSystemMatrix();
		SolveIncReflTrans();

		// Calc main fields in all the layers
		layers[0].U0 << inc, r;
		for (int i = 1; i < layers.size(); i++) {
			layers[i].U0 = (systemMatrices[i] * layers[0].U0.matrix()).array() + systemMatricesNL[i];
		}
		solved = true;
	}

	PowerFlows NonlinearTMM::GetPowerFlows() {
		if (!solved) {
			throw std::runtime_error("NonlinearTMM must be solved first.");
		}

		double omega = WlToOmega(wl);
		NonlinearLayer &l0 = layers[0];
		NonlinearLayer &lL = layers[layers.size() - 1];

		// Calc power flows
		double I, R, T;
		if (pol == P_POL) {
			double c = 1.0 / (2.0 * omega * constEps0);
			I = c * std::real(l0.hw.kz(F) / l0.eps) * std::norm(inc);
			R = c * std::real(l0.hw.kz(F) / l0.eps) * std::norm(r);
			T = c * std::real(lL.hw.kz(F) / lL.eps) * std::norm(t);
		}
		else if (pol == S_POL) {
			double c = 1.0 / (2.0 * omega * constMu0);
			I = c * std::real(l0.hw.kz(F)) * std::norm(inc);
			R = c * std::real(l0.hw.kz(F)) * std::norm(r);
			T = c * std::real(lL.hw.kz(F)) * std::norm(t);
		}
		else {
			throw std::runtime_error("Unknown polarization.");
		}

		PowerFlows res(inc, r, t, I, R, T);
		return res;
	}

	SweepResultNonlinearTMM * NonlinearTMM::Sweep(TMMParam param, const Eigen::Map<Eigen::ArrayXd>& values) {
		CheckPrerequisites(param);

		SweepResultNonlinearTMM *res = new SweepResultNonlinearTMM(values.size());

		#pragma omp parallel
		{
			// Make a copy of TMM
			NonlinearTMM tmmThread = *this;

			// Sweep
			#pragma omp for
			for (int i = 0; i < values.size(); i++) {
				tmmThread.SetParam(param, values(i));
				tmmThread.Solve();
				PowerFlows rr = tmmThread.GetPowerFlows();
				res->SetPowerFlows(i, rr);
			}
		}
		return res;
	}

	FieldsZ * NonlinearTMM::GetFields(const Eigen::Map<Eigen::ArrayXd>& zs, WaveDirection dir) {
		if (!solved) {
			throw std::runtime_error("NonlinearTMM must be solved first.");
		}

		// Find layerNr for every z
		Eigen::ArrayXi layerNrs(zs.size());
		Eigen::ArrayXd zInternal(zs.size());
		int curLayer = 0;
		double curLayerStartZ = 0.0;
		double curLayerEndZ = 0.0;
		for (int i = 0; i < zs.size(); i++) {
			double z = zs[i];

			// Error check
			if (i > 0 && z < zs[i]) {
				throw std::invalid_argument("zs array must be ordered");
			}

			// Move to correct layer
			while (z >= curLayerEndZ && curLayer + 1 < layers.size()) {
				curLayer++;
				curLayerStartZ = curLayerEndZ;
				curLayerEndZ += layers[curLayer].d;
			}
			layerNrs(i) = curLayer;
			zInternal(i) = z - curLayerStartZ;
		}

		// Allocate space (deletion is the responsibility of the caller!)
		FieldsZ *res = new FieldsZ(zs.size());

		// Top level parallelization is more efficient
		for (int i = 0; i < zs.size(); i++) {
			Fields f = layers[layerNrs(i)].GetFields(zInternal(i), dir);
			res->SetFieldsAtZ(i, f);
		}

		return res;
	}

	FieldsZX * NonlinearTMM::GetFields2D(const Eigen::Map<Eigen::ArrayXd>& zs, const Eigen::Map<Eigen::ArrayXd>& xs, WaveDirection dir) {
		if (!solved) {
			throw std::runtime_error("NonlinearTMM must be solved first.");
		}

		// Allocate space (deletion is the responsibility of the caller!)
		FieldsZX *res = new FieldsZX(zs.size(), xs.size(), pol);

		FieldsZ *F = GetFields(zs, dir);
		Eigen::ArrayXcd phaseX = (constI * layers[0].kx * xs).exp();
		res->SetFields(*F, phaseX, pol);
		delete F;
		return res;
	}

	double NonlinearTMM::GetAbsorbedPower() {
		if (!solved) {
			throw std::runtime_error("NonlinearTMM must be solved first.");
		}
		double res = 0.0;
		for (int i = 0; i < layers.size(); i++) {
			res += layers[i].GetAbsorbedPower();
		}
		return res;
	}
	
	void NonlinearTMM::SetParam(TMMParam param, bool value) {
		switch (param)
		{
		case PARAM_OVERRIDE_E0:
			overrideE0 = value;
			break;
		default:
			throw std::invalid_argument("Param not in list.");
			break;
		}
	}

	void NonlinearTMM::SetParam(TMMParam param, double value) {
		switch (param)
		{
		case PARAM_WL:
			wl = value;
			break;
		case PARAM_BETA:
			beta = value;
			break;
		case PARAM_I0:
			I0 = value;
			break;
		case PARAM_E0:
			E0 = (dcomplex)value;
			break;
		default:
			throw std::invalid_argument("Param not in list.");
			break;
		}
	}

	void NonlinearTMM::SetParam(TMMParam param, int value) {
		switch (param)
		{
		case PARAM_POL:
			pol = (Polarization)value;
			break;
		case PARAM_MODE:
			mode = (NonlinearTmmMode)value;
			break;
		default:
			throw std::invalid_argument("Param not in list.");
			break;
		}
	}

	void NonlinearTMM::SetParam(TMMParam param, dcomplex value) {
		switch (param)
		{
		case PARAM_E0:
			E0 = value;
			break;
		default:
			throw std::invalid_argument("Param not in list.");
			break;
		}
	}

	bool NonlinearTMM::GetBool(TMMParam param) {
		switch (param)
		{
		case PARAM_OVERRIDE_E0:
			return overrideE0;
			break;
		default:
			throw std::invalid_argument("Param not in list.");
			break;
		}
	}

	int NonlinearTMM::GetInt(TMMParam param) {
		switch (param)
		{
		case PARAM_POL:
			return (int)pol;
			break;
		case PARAM_MODE:
			return (int)mode;
			break;
		default:
			throw std::invalid_argument("Param not in list.");
			break;
		}
	}

	double NonlinearTMM::GetDouble(TMMParam param) {
		switch (param)
		{
		case PARAM_WL:
			return wl;
			break;
		case PARAM_BETA:
			return beta;
			break;
		case PARAM_I0:
			return I0;
			break;
		default:
			throw std::invalid_argument("Param not in list.");
			break;
		}
	}

	dcomplex NonlinearTMM::GetComplex(TMMParam param) {
		switch (param)
		{
		case PARAM_E0:
			return E0;
			break;
		default:
			throw std::invalid_argument("Param not in list.");
			break;
		}
	}

	
}