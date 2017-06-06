#include "NonlinearTmm.h"

namespace TMM {

	void OuterProductSSEEigenComplex(const ArrayXcd & X, const ArrayXcd & Y, MatrixXcd & R) {
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

	void OuterProductSSEEigenComplexAdd(const ArrayXcd & X, const ArrayXcd & Y, MatrixXcd & R) {
		// Identical to previous, but add to result (performance)
		register double* ptrR = (double*)&R(0, 0);
		const register dcomplex* ptrX = (dcomplex*)&X(0);
		const register dcomplex* ptrY = (dcomplex*)&Y(0);
		const register int n = X.size();
		const register int m = Y.size();

		for (int i = 0; i < n; i++) {
			register int k = 2 * i * m;
			register dcomplex v = ptrX[i];
			for (int j = 0; j < m; j++) {
				register int k2 = k + 2 * j;
				register dcomplex w = ptrY[j];
				register dcomplex r = multSSE(v, w);
				double realP = real(r);
				double imagP = imag(r);

				#pragma omp atomic
				ptrR[k2] += realP;
				#pragma omp atomic
				ptrR[k2 + 1] += imagP;
			}
		}
	}


	void OuterProductGoodEigenComplex(const ArrayXcd & X, const ArrayXcd & Y, MatrixXcd & R) {
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

	void OuterProductGoodEigenComplexAdd(const ArrayXcd & X, const ArrayXcd & Y, MatrixXcd & R) {
		double* ptrR = (double*)&R(0, 0);
		dcomplex* ptrX = (dcomplex*)&X(0);
		dcomplex* ptrY = (dcomplex*)&Y(0);
		int n = X.size();
		int m = Y.size();

		for (int i = 0; i < n; i++) {
			//int k = i * m;
			int k = 2 * i * m;
			dcomplex v = ptrX[i];
			for (int j = 0; j < m; j++) {
				dcomplex w = ptrY[j];
				//int k2 = k + j;
				int k2 = k + 2 * j;
				dcomplex r = v * w;
				
				#pragma omp atomic
				ptrR[k2] += real(r);
				#pragma omp atomic
				ptrR[k2 + 1] += imag(r);
			}
		}
	}

	void ThreadSafeMatrixAddNorm(MatrixXcd & mat, const MatrixXcd & toAdd) {
		if (mat.rows() != toAdd.rows() || mat.cols() != toAdd.cols()) {
			std::cerr << "mat and toAdd must be the same size." << std::endl;
			throw std::invalid_argument("mat and toAdd must be the same size.");
		}

		double* ptrMat = (double *)&mat(0, 0);
		dcomplex* ptrToAdd = (dcomplex *)&toAdd(0, 0);
		int n = mat.cols() * mat.rows();

		for (int i = 0; i < n; i++) {
			double norm = std::norm(ptrToAdd[i]);
			#pragma omp atomic	
			ptrMat[2 * i] += norm;
		}
	}

	Intensities::Intensities(dcomplex inc_, dcomplex r_, dcomplex t_, double I_, double R_, double T_) {
		inc = inc_;
		r = r_;
		t = t_;
		I = I_;
		R = R_;
		T = T_;
	}

	std::ostream & operator<<(std::ostream & os, const Intensities &p)
	{
		os << p.inc << " " << p.r << " " << p.t << " : " << p.I << " " << p.R << " " << p.T;
		return os;
	}

	SweepResultNonlinearTMM::SweepResultNonlinearTMM(int n, int outmask_, int layerNr_, double layerZ_) : inc(n), r(n), t(n), Ii(n), Ir(n), It(n), Ia(n), enh(n) {
		outmask = outmask_;
		layerNr = layerNr_;
		layerZ = layerZ_;
		inc.setConstant(constNAN);
		r.setConstant(constNAN);
		t.setConstant(constNAN);
		Ii.setConstant(constNAN);
		Ir.setConstant(constNAN);
		It.setConstant(constNAN);
		Ia.setConstant(constNAN);
		enh.setConstant(constNAN);
	}

	int SweepResultNonlinearTMM::GetOutmask() {
		return outmask;
	}

	void SweepResultNonlinearTMM::SetValues(int nr, NonlinearTMM & tmm) {

		if (outmask & SWEEP_PWRFLOWS) {
			Intensities pf = tmm.GetIntensities();
			inc(nr) = pf.inc;
			r(nr) = pf.r;
			t(nr) = pf.t;
			Ii(nr) = pf.I;
			Ir(nr) = pf.R;
			It(nr) = pf.T;
		}

		if (outmask & SWEEP_ABS) {
			double A_ = tmm.GetAbsorbedIntensity();
			Ia(nr) = A_;
		}

		if (outmask & SWEEP_ENH) {
			double enh_ = tmm.GetEnhancement(layerNr, layerZ);
			enh(nr) = enh_;
		}
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

	void FieldsZX::SetZero() {
		Ex.setZero();
		Ey.setZero();
		Ez.setZero();
		Hx.setZero();
		Hy.setZero();
		Hz.setZero();
	}

	Polarization FieldsZX::GetPol() {
		return pol;
	}

	void FieldsZX::SetFields(const FieldsZ & f, const ArrayXcd & phaseX, bool add) {
		int n = f.E.rows();
		int m = phaseX.size();

		// Select function
		OuterProductSSEEigenFunc func;
		if (add) {
			func = OuterProductSSEEigenComplexAdd;
			//func = OuterProductGoodEigenComplexAdd;
		}
		else {
			func = OuterProductSSEEigenComplex;
			//func = OuterProductGoodEigenComplex;
		}

		// Parallelisation has no/small effect
		switch (pol)
		{
		case TMM::P_POL:
			func(phaseX, f.E.col(0), Ex);
			func(phaseX, f.E.col(2), Ez);
			func(phaseX, f.H.col(1), Hy);
			// S-pol fields stay undefined because of performance
			break;
		case TMM::S_POL:
			func(phaseX, f.E.col(1), Ey);
			func(phaseX, f.H.col(0), Hx);
			func(phaseX, f.H.col(2), Hz);
			// P-pol fields stay undefined because of performance
			break;
		default:
			throw std::invalid_argument("Unknown polarization.");
			break;
		}
	}

	void FieldsZX::AddFields(const FieldsZ & f, const ArrayXcd & phaseX) {
		SetFields(f, phaseX, true);
	}

	void FieldsZX::AddSquaredFields(FieldsZX * toAdd) {
		switch (pol)
		{
		case TMM::P_POL:
			ThreadSafeMatrixAddNorm(Ex, toAdd->Ex);
			ThreadSafeMatrixAddNorm(Ez, toAdd->Ez);
			ThreadSafeMatrixAddNorm(Hy, toAdd->Hy);
			//Ex += toAdd->Ex.cwiseAbs2();
			//Ez += toAdd->Ez.cwiseAbs2();
			//Hy += toAdd->Hy.cwiseAbs2();
			// S-pol fields stay undefined because of performance
			break;
		case TMM::S_POL:
			ThreadSafeMatrixAddNorm(Ey, toAdd->Ey);
			ThreadSafeMatrixAddNorm(Hx, toAdd->Hx);
			ThreadSafeMatrixAddNorm(Hz, toAdd->Hz);
			//Ey += toAdd->Ey.cwiseAbs2();
			//Hx += toAdd->Hx.cwiseAbs2();
			//Hz += toAdd->Hz.cwiseAbs2();
			// P-pol fields stay undefined because of performance
			break;
		default:
			throw std::invalid_argument("Unknown polarization.");
			break;
		}
	}

	void FieldsZX::TakeSqrt() {
		switch (pol)
		{
		case TMM::P_POL:
			Ex = Ex.cwiseSqrt();
			Ez = Ez.cwiseSqrt();
			Hy = Hy.cwiseSqrt();
			break;
		case TMM::S_POL:
			Ey = Ey.cwiseSqrt();
			Hx = Hx.cwiseSqrt();
			Hz = Hz.cwiseSqrt();
			break;
		default:
			throw std::invalid_argument("Unknown polarization.");
			break;
		}
	}

	MatrixXd FieldsZX::GetENorm() {
		MatrixXd res;
		switch (pol)
		{
		case TMM::P_POL:
			res = (Ex.array().real().pow(2) + Ex.array().imag().pow(2) +
				Ez.array().real().pow(2) + Ez.array().imag().pow(2)).sqrt();
			break;
		case TMM::S_POL:
			res = (Ey.array().real().pow(2) + Ey.array().imag().pow(2)).sqrt();
			break;
		default:
			std::cerr << "Unknown polarization." << std::endl;
			throw std::invalid_argument("Unknown polarization.");
			break;
		}
		return res;
	}

	void NonlinearTMM::CheckPrerequisites(TMMParam toIgnore) {
		// Check params
		if (mode != MODE_NONLINEAR) {
			if (toIgnore != PARAM_WL && isnan(wl)) {
				throw std::invalid_argument("Wavelength is not set.");
			}

			if (toIgnore != PARAM_BETA && isnan(beta)) {
				throw std::invalid_argument("Beta is not set.");
			}
		}

		if (toIgnore != PARAM_POL && pol == NOT_DEFINED_POL) {
			throw std::invalid_argument("Polarization is not set.");
		}

		if (layers.size() < 2) {
			throw std::invalid_argument("TMM must have at least 2 layers.");
		}

		if (mode == MODE_VACUUM_FLUCTUATIONS) {
			if (isnan(deltaWlSpdc)) {
				std::cerr << "No value for deltaWlSpdc" << std::endl;
				throw std::invalid_argument("No value for deltaWlSpdc");
			}

			if (isnan(solidAngleSpdc)) {
				std::cerr << "No value for solidAngleSpdc" << std::endl;
				throw std::invalid_argument("No value for solidAngleSpdc");
			}

			if (isnan(deltaThetaSpdc)) {
				std::cerr << "No value for deltaThetaSpdc" << std::endl;
				throw std::invalid_argument("No value for deltaThetaSpdc");
			}

			if (isnan(wlP1Spdc)) {
				std::cerr << "No value for wlP1Spdc" << std::endl;
				throw std::invalid_argument("No value for wlP1Spdc");
			}

			if (isnan(betaP1Spdc)) {
				std::cerr << "No value for betaP1Spdc" << std::endl;
				throw std::invalid_argument("No value for betaP1Spdc");
			}
		}
	}

	// Setters

	void NonlinearTMM::SetWl(double wl_) {
		wl = wl_;
	}

	void NonlinearTMM::SetBeta(double beta_) {
		beta = beta_;
	}

	void NonlinearTMM::SetPolarization(Polarization pol_) {
		pol = pol_;
	}

	void NonlinearTMM::SetI0(double I0_) {
		I0 = I0_;
	}

	void NonlinearTMM::SetOverrideE0(bool overrideE0_) {
		overrideE0 = overrideE0_;
	}

	void NonlinearTMM::SetE0(dcomplex E0_) {
		E0 = E0_;
	}

	void NonlinearTMM::SetMode(NonlinearTmmMode mode_) {
		mode = mode_;
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
		if (mode == MODE_INCIDENT || mode == MODE_VACUUM_FLUCTUATIONS) {
			if (mode == MODE_INCIDENT && !overrideE0) {
				// Intensity given by I0 (intensity at normal incidence), I = I0 * cos(th0)				
				double cosTh0 = real(l0.hw.GetKzF()) / real(l0.k);
				if (pol == P_POL) {
					inc = std::sqrt(2.0 * omega * constEps0 / std::real(l0.hw.kz(F)) * real(l0.eps) * I0 * cosTh0);
				}
				else if (pol == S_POL) {
					inc = std::sqrt(2.0 * omega * constMu0 / std::real(l0.hw.kz(F)) * I0 * cosTh0);
				}
				else {
					throw std::runtime_error("Unknown polarization");
				}
			}
			else {
				// Pump wave E0 is given in vacuum
				dcomplex E0This = E0;
				if (mode == MODE_VACUUM_FLUCTUATIONS) {
					E0This = CalcVacFuctuationsE0();
				}

				if (pol == P_POL) {
					inc = E0This * std::sqrt(real(l0.n)) * constEps0 * constC;
				}
				else if (pol == S_POL) {
					inc = E0This / std::sqrt(real(l0.n));
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

	void NonlinearTMM::SolveWave(ArrayXd * betas, ArrayXcd * E0s) {
		if (wave.GetWaveType() == SPDCWAVE && mode != MODE_VACUUM_FLUCTUATIONS) {
			std::cerr << "NonlinearTMM must be in MODE_VACUUM_FLUCTUATIONS mode to use SPDC wave." << std::endl;
			throw std::invalid_argument("NonlinearTMM must be in MODE_VACUUM_FLUCTUATIONS mode to use SPDC wave.");
		}

		Material *matLayerF = layers[0].GetMaterial();
		Material *matLayerL = NULL;
		double deltaKxSpdc = CalcDeltaKxSpdc();
		wave.Solve(wl, beta, matLayerF, matLayerL, deltaKxSpdc);
		*betas = wave.GetBetas();
		*E0s = wave.GetExpansionCoefsKx();
	}

	double NonlinearTMM::CalcVacFuctuationsE0() {
		double wlP2 = wl;
		double wlGen = OmegaToWl(WlToOmega(wlP1Spdc) - WlToOmega(wlP2));
		double omegaP1 = WlToOmega(wlP1Spdc);
		double omegaP2 = WlToOmega(wlP2);
		double omegaGen = WlToOmega(wlGen);
		// Calc vacuum fluctuations stength
		double ESqr = (deltaWlSpdc * solidAngleSpdc / deltaThetaSpdc) *
			(constHbar / (8.0 * constEps0 * std::pow(PI, 4))) *
			(std::pow(omegaGen, 3) * std::pow(omegaP2, 2) / (pow(constC, 4))) *
			(constC / omegaP2);
		double EVac = std::sqrt(ESqr);
		return EVac;
	}

	double NonlinearTMM::CalcDeltaKxSpdc() {
		// TODO: restrict ublic usage

		// Calc deltaKx
		if (wave.GetWaveType() == SPDCWAVE) {
			double wlP2 = wl;
			double betaP2 = beta;
			double wlGen = OmegaToWl(WlToOmega(wlP1Spdc) - WlToOmega(wlP2));
			double betaGen = wlGen * (betaP1Spdc / wlP1Spdc - betaP2 / wlP2);
			Material *matLayer0 = GetLayer(0)->GetMaterial();
			double n0 = real(matLayer0->GetN(wlGen));
			double kz0 = 2.0 * PI / wlGen * std::sqrt(n0 * n0 - betaGen * betaGen);
			double res = kz0 / (2.0 * n0) * deltaThetaSpdc;
			return res;
		}

		// No SPDC wave involved, return NAN
		return constNAN;
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
		deltaWlSpdc = constNAN;
		solidAngleSpdc = constNAN;
		deltaThetaSpdc = constNAN;
		wlP1Spdc = constNAN;
		betaP1Spdc = constNAN;
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

		// Check first layer
		if (beta >= real(layers[0].n)) {
			std::cerr << "Light cannot propagate in the first medium." << std::endl;
			throw std::invalid_argument("Light cannot propagate in the first medium.");
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

	Intensities NonlinearTMM::GetIntensities() const {
		if (!solved) {
			throw std::runtime_error("NonlinearTMM must be solved first.");
		}

		double omega = WlToOmega(wl);
		const NonlinearLayer &l0 = layers[0];
		const NonlinearLayer &lL = layers[layers.size() - 1];

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

		Intensities res(inc, r, t, I, R, T);
		return res;
	}

	SweepResultNonlinearTMM * NonlinearTMM::Sweep(TMMParam param, const Eigen::Map<ArrayXd>& values, int outmask, int paramLayer, int layerNr, double layerZ) {
		CheckPrerequisites(param);
		if (layerNr < 0 || layerNr > layers.size()) {
			throw std::invalid_argument("Invalid layer index.");
		}

		SweepResultNonlinearTMM *res = new SweepResultNonlinearTMM(values.size(), outmask, layerNr, layerZ);

		#pragma omp parallel
		{
			// Make a copy of TMM
			NonlinearTMM tmmThread = *this;

			// Sweep
			#pragma omp for
			for (int i = 0; i < values.size(); i++) {
				tmmThread.SetParam(param, values(i), paramLayer);
				tmmThread.Solve();
				res->SetValues(i, tmmThread);
			}
		}
		return res;
	}

	FieldsZ * NonlinearTMM::GetFields(const Eigen::Map<ArrayXd>& zs, WaveDirection dir) {
		if (!solved) {
			throw std::runtime_error("NonlinearTMM must be solved first.");
		}

		// Find layerNr for every z
		Eigen::ArrayXi layerNrs(zs.size());
		ArrayXd zInternal(zs.size());
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

	FieldsZX * NonlinearTMM::GetFields2D(const Eigen::Map<ArrayXd>& zs, const Eigen::Map<ArrayXd>& xs, WaveDirection dir) {
		if (!solved) {
			throw std::runtime_error("NonlinearTMM must be solved first.");
		}

		// Allocate space (deletion is the responsibility of the caller!)
		FieldsZX *res = new FieldsZX(zs.size(), xs.size(), pol);
		double kx = layers[0].kx;

		FieldsZ *f = GetFields(zs, dir);
		ArrayXcd phaseX = (constI * kx * xs).exp();
		res->SetFields(*f, phaseX);
		delete f;
		return res;
	}

	FieldsZX * NonlinearTMM::WaveGetFields2D(const Eigen::Map<ArrayXd>& zs, const Eigen::Map<ArrayXd>& xs, WaveDirection dir) {
		CheckPrerequisites();

		if (mode == MODE_NONLINEAR) {
			throw std::runtime_error("For nonlinear mode use the method of SecondOrderNLTMM");
		}

		// Solve wave
		ArrayXd betas;
		ArrayXcd E0s;
		SolveWave(&betas, &E0s);
		ArrayXd kxs = betas * 2.0 * PI / wl;

		// Allocate space (deletion is the responsibility of the caller!)
		FieldsZX *res = new FieldsZX(zs.size(), xs.size(), pol);
		res->SetZero(); // We are summing up

		if (wave.IsCoherent()) {
			// Coherent
			#pragma omp parallel
			{
				NonlinearTMM tmmThread = *this;
				tmmThread.SetOverrideE0(true);
				#pragma omp for
				for (int i = 0; i < betas.size(); i++) {

					// Solve TMM
					tmmThread.SetBeta(betas(i));
					tmmThread.SetE0(E0s(i));
					tmmThread.Solve();

					// Integrate fields
					double dkx = GetDifferential(kxs, i);
					FieldsZ *f = tmmThread.GetFields(zs, dir);
					ArrayXcd phaseX = (constI * kxs(i) * xs).exp() * dkx;
					res->AddFields(*f, phaseX);
					delete f;
				}
			}
		}
		else {
			// Incoherent
			#pragma omp parallel
			{
				NonlinearTMM tmmThread = *this;
				tmmThread.SetOverrideE0(true);
				#pragma omp for
				for (int i = 0; i < betas.size(); i++) {

					// Solve TMM
					tmmThread.SetBeta(betas(i));
					tmmThread.SetE0(E0s(i));
					tmmThread.Solve();

					// Integrate fields
					double dkx = GetDifferential(kxs, i);
					
					FieldsZ *f = tmmThread.GetFields(zs, dir);
					ArrayXcd phaseX = (constI * kxs(i) * xs).exp() * dkx;
					FieldsZX *coherentFields = new FieldsZX(zs.size(), xs.size(), pol);
					coherentFields->SetZero();
					coherentFields->AddFields(*f, phaseX);
					res->AddSquaredFields(coherentFields);
					delete coherentFields;
					delete f;
				}
			}
			res->TakeSqrt();
		}

		return res;
	}

	double NonlinearTMM::WaveGetEnhancement(int layerNr, double z) {
		if (layerNr < 0 || layerNr > layers.size()) {
			throw std::invalid_argument("Invalid layer index.");
		}

		double layerZ = 0.0;
		for (int i = 1; i < layerNr; i++) {
			layerZ += layers[i].d;
		}
		layerZ += z;

		double xs[] = { 0.0 };
		double zs0[] = { -1e-9 };
		double zsL[] = { layerZ };
		Eigen::Map<ArrayXd> xsMap(xs, 1);
		Eigen::Map<ArrayXd> zs0Map(zs0, 1);
		Eigen::Map<ArrayXd> zsLMap(zsL, 1);

		FieldsZX *f0 = WaveGetFields2D(zs0Map, xsMap, F);
		FieldsZX *fL = WaveGetFields2D(zsLMap, xsMap, TOT);

		double EN0 = f0->GetENorm()(0, 0);
		double ENL = fL->GetENorm()(0, 0);
		delete f0;
		delete fL;
		double n0 = real(layers[0].GetMaterial()->GetN(wl));
		double res = ENL / (EN0 * std::sqrt(n0));
		return res;
	}

	WaveSweepResultNonlinearTMM * NonlinearTMM::WaveSweep(TMMParam param, const Eigen::Map<ArrayXd>& values, int outmask, int paramLayer, int layerNr, double layerZ) {
		CheckPrerequisites(param);
		if (layerNr < 0 || layerNr > layers.size()) {
			throw std::invalid_argument("Invalid layer index.");
		}

		WaveSweepResultNonlinearTMM *res = new WaveSweepResultNonlinearTMM(values.size(), outmask, layerNr, layerZ);
		#pragma omp parallel
		{
			// Make a copy of TMM
			NonlinearTMM tmmThread = *this;

			// Sweep
			#pragma omp for
			for (int i = 0; i < values.size(); i++) {
				tmmThread.SetParam(param, values(i), paramLayer);
				res->SetValues(i, tmmThread);
			}
		}
		return res;
	}

	double NonlinearTMM::GetAbsorbedIntensity() const{
		if (!solved) {
			throw std::runtime_error("NonlinearTMM must be solved first.");
		}
		double res = 0.0;
		for (int i = 0; i < layers.size(); i++) {
			res += layers[i].GetAbsorbedIntensity();
		}
		return res;
	}

	double NonlinearTMM::GetEnhancement(int layerNr, double z) {
		if (!solved) {
			throw std::runtime_error("NonlinearTMM must be solved first.");
		}

		if (layerNr < 0 || layerNr > layers.size()) {
			throw std::invalid_argument("Invalid layer index.");
		}

		double ENL = layers[layerNr].GetENorm(z, TOT);
		double EN0 = layers[0].GetENorm(z, F) * std::sqrt(real(layers[0].n)); // Electrical field in vacuum
		double res = ENL / EN0;
		return res;

	}

	Wave * NonlinearTMM::GetWave() {
		return &wave;
	}

	pairdd NonlinearTMM::WaveGetPowerFlows(int layerNr, double x0, double x1, double z) {
		CheckPrerequisites();
		if (mode == MODE_NONLINEAR) {
			std::cerr << "For nonlinear mode use the method of SecondOrderNLTMM" << std::endl;
			throw std::runtime_error("For nonlinear mode use the method of SecondOrderNLTMM");
		}

		if (layerNr < 0 || layerNr > layers.size()) {
			throw std::invalid_argument("Invalid layer index.");
		}

		// Solve wave
		ArrayXd betas;
		ArrayXcd E0s;
		SolveWave(&betas, &E0s);
		double Ly = wave.GetLy();
		
		// Adjust range
		pairdd xrange = wave.GetXRange();
		if (isnan(x0)) {
			x0 = xrange.first;
		}

		if (isnan(x1)) {
			x1 = xrange.second;
		}

		// Init memory
		Eigen::MatrixX2cd Us(betas.size(), 2);
		Eigen::MatrixX2cd kzs(betas.size(), 2);

		// Solve for every beta
		bool oldOverrideE0 = GetOverrideE0();
		double oldBeta = GetBeta();
		SetOverrideE0(true);
		for (int i = 0; i < betas.size(); i++) {
			SetBeta(betas(i));
			SetE0(E0s(i));
			Solve();
			Us.row(i) = layers[layerNr].GetMainFields(0.0);
			kzs.row(i) = layers[layerNr].hw.kz;
		}
		SetOverrideE0(oldOverrideE0);
		SetBeta(oldBeta);

		// Integrate powers

		ArrayXd kxs = betas * layers[0].k0;
		dcomplex epsLayer = layers[layerNr].eps;
		pairdd res(0.0, 0.0);
		if (wave.IsCoherent()) {
			// Coherent
			res = IntegrateWavePower(layerNr, pol, wl, epsLayer, Us, kxs, kzs, x0, x1, z, Ly);
		}
		else {
			// Incoherent
			for (int i = 0; i < betas.size(); i++) {
				pairdd r = IntegrateWavePower(layerNr, pol, wl, epsLayer, Us.row(i), kxs.row(i), kzs.row(i), x0, x1, z, Ly);
				double dkx = GetDifferential(kxs, i);
				res.first += r.first * dkx;
				res.second += r.second * dkx;
			}
		}
		return res;
	}
	
	void NonlinearTMM::SetParam(TMMParam param, double value, int paramLayer) {
		if (GetParamType(param) == PTYPE_NONLINEAR_TMM) {
			switch (param)
			{
			case PARAM_WL:
				SetWl(value);
				break;
			case PARAM_BETA:
				SetBeta(value);
				break;
			case PARAM_I0:
				SetI0(value);
				break;
			case PARAM_E0:
				SetE0(value);
				break;
			default:
				std::cerr << "Param not in list." << std::endl;
				throw std::invalid_argument("Param not in list.");
				break;
			}
		}
		else if (GetParamType(param) == PTYPE_NONLINEAR_LAYER) {
			if (paramLayer < 0 || paramLayer >= layers.size()) {
				std::cerr << "Invalid layer number." << std::endl;
				throw std::invalid_argument("Invalid layer number.");
			}
			layers[paramLayer].SetParam(param, value);
		}
		else if (GetParamType(param) == PTYPE_WAVE) {
			wave.SetParam(param, value);
		}
		else {
			std::cerr << "Invalid param type" << std::endl;
			throw std::invalid_argument("Invalid param type");
		}
	}

	void NonlinearTMM::SetParam(TMMParam param, dcomplex value, int paramLayer) {
		switch (param)
		{
		case PARAM_E0:
			SetE0(value);
			break;
		default:
			std::cerr << "Param not in list." << std::endl;
			throw std::invalid_argument("Param not in list.");
			break;
		}
	}

	void NonlinearTMM::UpdateSPDCParams(double deltaWlSpdc_, double solidAngleSpdc_, double deltaThetaSpdc_, double wlP1Spdc_, double betaP1Spdc_) {
		deltaWlSpdc = deltaWlSpdc_;
		solidAngleSpdc = solidAngleSpdc_;
		deltaThetaSpdc = deltaThetaSpdc_;
		wlP1Spdc = wlP1Spdc_;
		betaP1Spdc = betaP1Spdc_;
	}

	// Getters

	double NonlinearTMM::GetWl() {
		return wl;
	}

	double NonlinearTMM::GetBeta() {
		return beta;
	}

	Polarization NonlinearTMM::GetPolarization() {
		return pol;
	}

	double NonlinearTMM::GetI0() {
		return I0;
	}

	bool NonlinearTMM::GetOverrideE0() {
		return overrideE0;
	}

	dcomplex NonlinearTMM::GetE0() {
		return E0;
	}

	NonlinearTmmMode NonlinearTMM::GetMode() {
		return mode;
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
			std::cerr << "Param not in list." << std::endl;
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
			std::cerr << "Param not in list." << std::endl;
			throw std::invalid_argument("Param not in list.");
			break;
		}
	}

	
	WaveSweepResultNonlinearTMM::WaveSweepResultNonlinearTMM(int n, int outmask_, int layerNr_, double layerZ_) : Pi(n), Pr(n), Pt(n), enh(n), beamArea(n) {
		outmask = outmask_;
		layerNr = layerNr_;
		layerZ = layerZ_;
		Pi.setConstant(constNAN);
		Pr.setConstant(constNAN);
		Pt.setConstant(constNAN);
		enh.setConstant(constNAN);
		beamArea.setConstant(constNAN);
	}

	int WaveSweepResultNonlinearTMM::GetOutmask() {
		return outmask;
	}

	void WaveSweepResultNonlinearTMM::SetValues(int nr, NonlinearTMM & tmm) {
		// First layer
		if ((outmask & SWEEP_I) || (outmask & SWEEP_R)) {
			pairdd pf0 = tmm.WaveGetPowerFlows(0);
			Pi(nr) = pf0.first;
			Pr(nr) = pf0.second;
		}

		// Last layer
		if (outmask & SWEEP_T) {
			pairdd pfL = tmm.WaveGetPowerFlows(tmm.LayersCount() - 1);
			Pt(nr) = pfL.first;
		}

		if (outmask & SWEEP_ENH) {
			double enh_ = tmm.WaveGetEnhancement(layerNr, layerZ);
			enh(nr) = enh_;
		}

		beamArea(nr) = tmm.GetWave()->GetBeamArea();
	}

}