#include "SecondOrderNLTMM.h"

namespace TMM {

	SweepResultSecondOrderNLTMM::SweepResultSecondOrderNLTMM(int n, int outmask, int layerNr_, double layerZ_) : 
		P1(n, outmask, layerNr_, layerZ_), P2(n, outmask, layerNr_, layerZ_), Gen(n, outmask, layerNr_, layerZ_) {}

	void SweepResultSecondOrderNLTMM::SetValues(int nr, SecondOrderNLTMM & tmm) {
		P1.SetValues(nr, *tmm.GetP1());
		P2.SetValues(nr, *tmm.GetP2());
		Gen.SetValues(nr, *tmm.GetGen());
	}

	void SecondOrderNLTMM::UpdateGenParams()
	{
		double wlP1 = tmmP1.GetDouble(PARAM_WL);
		double wlP2 = tmmP2.GetDouble(PARAM_WL);
		double betaP1 = tmmP1.GetDouble(PARAM_BETA);
		double betaP2 = tmmP2.GetDouble(PARAM_BETA);

		switch (process)
		{
		case TMM::SHG:
			throw std::runtime_error("Not implemented");
			break;
		case TMM::SFG:
			wlGen = OmegaToWl(WlToOmega(wlP1) + WlToOmega(wlP2));
			betaGen = wlGen * (betaP1 / wlP1 + betaP2 / wlP2);
			break;
		case TMM::DFG:
			wlGen = OmegaToWl(WlToOmega(wlP1) - WlToOmega(wlP2));
			betaGen = wlGen * (betaP1 / wlP1 - betaP2 / wlP2);
			break;
		default:
			throw std::runtime_error("Unknown process.");
			break;
		}
	}
	void SecondOrderNLTMM::CalcInhomogeneosWaveParams(int layerNr, Material *material, InhomogeneosWaveParams * kpS, InhomogeneosWaveParams * kpA)
	{
		dcomplex kzFP1 = tmmP1.GetLayer(layerNr)->GetHw()->GetKz()(F);
		dcomplex kzFP2 = tmmP2.GetLayer(layerNr)->GetHw()->GetKz()(F);
		Fields FP1F = tmmP1.GetLayer(layerNr)->GetFields(0.0, F);
		Fields FP2F = tmmP2.GetLayer(layerNr)->GetFields(0.0, F);
		Fields FP1B = tmmP1.GetLayer(layerNr)->GetFields(0.0, B);
		Fields FP2B = tmmP2.GetLayer(layerNr)->GetFields(0.0, B);
		Chi2Tensor &chi2 = material->chi2;

		switch (process)
		{
		case TMM::SHG:
			throw std::runtime_error("Not implemented");
			break;
		case TMM::SFG:
			kpS->kSzF = kzFP1 + kzFP2;
			kpA->kSzF = kzFP1 - kzFP2;
			kpS->pF = chi2.GetNonlinearPolarization(FP1F.E, FP2F.E);
			kpS->pB = chi2.GetNonlinearPolarization(FP1B.E, FP2B.E);
			kpA->pF = chi2.GetNonlinearPolarization(FP1F.E, FP2B.E);
			kpA->pB = chi2.GetNonlinearPolarization(FP1B.E, FP2F.E);
			break;
		case TMM::DFG:
			kpS->kSzF = kzFP1 - std::conj(kzFP2);
			kpA->kSzF = kzFP1 + std::conj(kzFP2);
			kpS->pF = chi2.GetNonlinearPolarization(FP1F.E, FP2F.E.conjugate());
			kpS->pB = chi2.GetNonlinearPolarization(FP1B.E, FP2B.E.conjugate());
			kpA->pF = chi2.GetNonlinearPolarization(FP1F.E, FP2B.E.conjugate());
			kpA->pB = chi2.GetNonlinearPolarization(FP1B.E, FP2F.E.conjugate());
			break;
		default:
			throw std::runtime_error("Unknown process.");
			break;
		}

		// TODO:
		if (imag(kpS->kSzF) < 0.0) {
			throw std::runtime_error("not implemented");
		}

		if (imag(kpA->kSzF) < 0.0) {
			throw std::runtime_error("not implemented");
		}
	}

	void SecondOrderNLTMM::SolveFundamentalFields() {
		tmmP1.Solve();
		tmmP2.Solve();
	}
	void SecondOrderNLTMM::SolveGeneratedField() {
		tmmGen.SetParam(PARAM_WL, wlGen);
		tmmGen.SetParam(PARAM_BETA, betaGen);
		
		// Insert nonlinearities
		for (int i = 0; i < tmmGen.LayersCount(); i++) {
			Material *material = tmmGen.GetLayer(i)->GetMaterial();
			if (material->IsNonlinear()) {
				InhomogeneosWaveParams kpS, kpA;
				CalcInhomogeneosWaveParams(i, material, &kpS, &kpA);
				tmmGen.GetLayer(i)->SetNonlinearity(kpS, kpA);
			}
			else {
				tmmGen.GetLayer(i)->ClearNonlinearity();
			}
		}

		// Solve generated fields
		tmmGen.Solve();
	}
	SecondOrderNLTMM::SecondOrderNLTMM()
	{
		SetProcess(SFG);
		tmmP1.SetParam(PARAM_MODE, MODE_INCIDENT);
		tmmP2.SetParam(PARAM_MODE, MODE_INCIDENT);
		tmmGen.SetParam(PARAM_MODE, MODE_NONLINEAR);
	}

	void SecondOrderNLTMM::SetProcess(NonlinearProcess process_)
	{
		process = process_;
	}

	void SecondOrderNLTMM::AddLayer(double d_, Material *material_)
	{
		tmmP1.AddLayer(d_, material_);
		tmmP2.AddLayer(d_, material_);
		tmmGen.AddLayer(d_, material_);
	}

	void SecondOrderNLTMM::Solve()
	{
		UpdateGenParams();
		SolveFundamentalFields();
		SolveGeneratedField();
	}

	SecondOrderNLPowerFlows SecondOrderNLTMM::GetPowerFlows() {
		SecondOrderNLPowerFlows res;
		res.P1 = tmmP1.GetPowerFlows();
		res.P2 = tmmP2.GetPowerFlows();
		res.Gen = tmmGen.GetPowerFlows();
		return res;
	}

	NonlinearTMM * SecondOrderNLTMM::GetP1() {
		return &tmmP1;
	}

	NonlinearTMM * SecondOrderNLTMM::GetP2() {
		return &tmmP2;
	}

	NonlinearTMM * SecondOrderNLTMM::GetGen() {
		return &tmmGen;
	}

	SweepResultSecondOrderNLTMM * SecondOrderNLTMM::Sweep(TMMParam param, const Eigen::Map<Eigen::ArrayXd>& valuesP1, const Eigen::Map<Eigen::ArrayXd>& valuesP2, int outmask, int layerNr, double layerZ) {
		if (valuesP1.size() != valuesP2.size()) {
			throw std::invalid_argument("Value arrays must have the same size.");
		}

		tmmP1.CheckPrerequisites(param);
		tmmP2.CheckPrerequisites(param);
		tmmGen.CheckPrerequisites(param);

		// Alloc memory for result (dealloc is responsibility of the user)
		SweepResultSecondOrderNLTMM *res = new SweepResultSecondOrderNLTMM(valuesP1.size(), outmask, layerNr, layerZ);

		#pragma omp parallel
		{
			//int this_thread = omp_get_thread_num();
			//std::cout << "Thread started: " << this_thread << std::endl;

			// Make a copy
			SecondOrderNLTMM nlTMMCopy = *this;

			// Sweep
			#pragma omp for
			for (int i = 0; i < valuesP1.size(); i++) {
				//Set sweep param
				nlTMMCopy.GetP1()->SetParam(param, valuesP1(i));
				nlTMMCopy.GetP2()->SetParam(param, valuesP2(i));

				// Solver
				nlTMMCopy.Solve();
				res->SetValues(i, nlTMMCopy);
			}
		}
		return res;
	}

	FieldsZX * SecondOrderNLTMM::GetGenWaveFields2D(const Eigen::Map<Eigen::ArrayXd>& betasP1, const Eigen::Map<Eigen::ArrayXd>& betasP2, const Eigen::Map<Eigen::ArrayXcd>& E0sP1, const Eigen::Map<Eigen::ArrayXcd>& E0sP2, const Eigen::Map<Eigen::ArrayXd>& zs, const Eigen::Map<Eigen::ArrayXd>& xs, WaveDirection dir) {
		// Do checking
		tmmP1.CheckPrerequisites(PARAM_BETA);
		tmmP2.CheckPrerequisites(PARAM_BETA);
		tmmGen.CheckPrerequisites(PARAM_BETA);

		if (E0sP1.size() != betasP1.size()) {
			throw std::invalid_argument("Arrays (betasP1, E0sP1) must have the same length.");
		}

		if (E0sP2.size() != betasP2.size()) {
			throw std::invalid_argument("Arrays (betasP2, E0sP2) must have the same length.");
		}

		// kxs
		Eigen::ArrayXd kxsP1 = betasP1 * 2.0 * PI / tmmP1.GetDouble(PARAM_WL);
		Eigen::ArrayXd kxsP2 = betasP2 * 2.0 * PI / tmmP2.GetDouble(PARAM_WL);

		// Allocate space (deletion is the responsibility of the caller!)
		FieldsZX *res = new FieldsZX(zs.size(), xs.size(), (Polarization)tmmGen.GetInt(PARAM_POL));
		res->SetZero();

		#pragma omp parallel
		{
			SecondOrderNLTMM tmmThread = *this;
			tmmThread.tmmP1.SetParam(PARAM_OVERRIDE_E0, true);
			tmmThread.tmmP2.SetParam(PARAM_OVERRIDE_E0, true);
			#pragma omp for
			for (int i = 0; i < betasP1.size(); i++) {
				tmmThread.tmmP1.SetParam(PARAM_BETA, betasP1(i));
				tmmThread.tmmP1.SetParam(PARAM_E0, E0sP1(i));
				double dkxP1 = GetDifferential(kxsP1, i);
				for (int j = 0; j < betasP2.size(); j++) {
					tmmThread.tmmP2.SetParam(PARAM_BETA, betasP2(j));
					tmmThread.tmmP2.SetParam(PARAM_E0, E0sP2(j));
					double dkxP2 = GetDifferential(kxsP2, j);

					// Solve
					tmmThread.Solve();

					// Integrate fields
					double kxGen = tmmThread.tmmGen.GetLayer(0)->GetKx();
					FieldsZ *fGen = tmmThread.tmmGen.GetFields(zs, dir);
					Eigen::ArrayXcd phaseXGen = (constI * kxGen * xs).exp() * dkxP1 * dkxP2;
					res->AddFields(*fGen, phaseXGen);
					delete fGen;
				}
			}
		}
		return res;
	}

	pairdd SecondOrderNLTMM::GetPowerFlowsGenForWave(const Eigen::Map<Eigen::ArrayXd>& betasP1, const Eigen::Map<Eigen::ArrayXd>& betasP2, const Eigen::Map<Eigen::ArrayXcd>& E0sP1, const Eigen::Map<Eigen::ArrayXcd>& E0sP2, int layerNr, double x0, double x1, double z, double Ly, WaveDirection dir) {
		// Do checking

		if (E0sP1.size() != betasP1.size()) {
			throw std::invalid_argument("Arrays (betasP1, E0sP1) must have the same length.");
		}

		if (E0sP2.size() != betasP2.size()) {
			throw std::invalid_argument("Arrays (betasP2, E0sP2) must have the same length.");
		}

		if (layerNr < 0 || layerNr > tmmP1.LayersCount()) {
			throw std::invalid_argument("Invalid layer index.");
		}

		// Init memory
		int nTot = betasP1.size() * betasP2.size();
		Eigen::MatrixX2cd UsUnsorted(nTot, 2);
		Eigen::MatrixX2cd kzsUnsorted(nTot, 2);
		Eigen::ArrayXd kxsUnsorted(nTot);

		// Solve for every beta
		bool oldOverrideE0P1 = tmmP1.GetBool(PARAM_OVERRIDE_E0);
		bool oldOverrideE0P2 = tmmP2.GetBool(PARAM_OVERRIDE_E0);
		tmmP1.SetParam(PARAM_OVERRIDE_E0, true);
		tmmP2.SetParam(PARAM_OVERRIDE_E0, true);
		for (int i = 0; i < betasP1.size(); i++) {
			tmmP1.SetParam(PARAM_BETA, betasP1(i));
			tmmP1.SetParam(PARAM_E0, E0sP1(i));
			for (int j = 0; j < betasP2.size(); j++) {
				tmmP2.SetParam(PARAM_BETA, betasP2(j));
				tmmP2.SetParam(PARAM_E0, E0sP2(j));
				Solve();

				int index = i * betasP2.size() + j;
				UsUnsorted.row(index) = tmmGen.GetLayer(layerNr)->GetMainFields(0.0);
				kzsUnsorted.row(index) = tmmGen.GetLayer(layerNr)->GetHw()->GetKz();
				kxsUnsorted(index) = tmmGen.GetLayer(layerNr)->GetKx();
			}
		}
		tmmP1.SetParam(PARAM_OVERRIDE_E0, oldOverrideE0P1);
		tmmP2.SetParam(PARAM_OVERRIDE_E0, oldOverrideE0P2);

		// Sort values by kxs
		Eigen::ArrayXi indices(nTot);
		for (int i = 0; i < nTot; i++) {
			indices(i) = i;
		}

		std::sort(indices.data(), indices.data() + nTot, [&kxsUnsorted](const int a, int b) -> bool
		{
			return kxsUnsorted(a) < kxsUnsorted(b);
		});

		Eigen::MatrixX2cd Us(nTot, 2);
		Eigen::MatrixX2cd kzs(nTot, 2);
		Eigen::ArrayXd kxs(nTot);

		for (int i = 0; i < nTot; i++) {
			Us.row(i) = UsUnsorted.row(indices(i));
			kzs.row(i) = kzsUnsorted.row(indices(i));
			kxs(i) = kxsUnsorted(indices(i));
		}

		// Integrate powers
		double PF = constNAN, PB = constNAN;
		double wlGen = tmmGen.GetDouble(PARAM_WL);
		Polarization polGen = (Polarization)tmmGen.GetInt(PARAM_POL);
		dcomplex epsLayer0 = tmmGen.GetLayer(0)->eps;

		switch (dir)
		{
		case TMM::F:
			PF = IntegrateWavePower(layerNr, polGen, wlGen, epsLayer0, Us.col(F), kxs, kzs.col(F), x0, x1, z, Ly);
			break;
		case TMM::B:
			PB = -IntegrateWavePower(layerNr, polGen, wlGen, epsLayer0, Us.col(B), kxs, kzs.col(B), x0, x1, z, Ly);
			break;
		case TMM::TOT:
			PF = IntegrateWavePower(layerNr, polGen, wlGen, epsLayer0, Us.col(F), kxs, kzs.col(F), x0, x1, z, Ly);
			PB = -IntegrateWavePower(layerNr, polGen, wlGen, epsLayer0, Us.col(B), kxs, kzs.col(B), x0, x1, z, Ly);
			break;
		default:
			throw std::invalid_argument("Invalid direction.");
			break;
		}
		return pairdd(PF, PB);
	}

}