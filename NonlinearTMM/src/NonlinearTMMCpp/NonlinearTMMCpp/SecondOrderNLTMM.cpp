#include "SecondOrderNLTMM.h"

namespace TMM {

	SweepResultSecondOrderNLTMM::SweepResultSecondOrderNLTMM(int n, int outmask_, int layerNr_, double layerZ_) : 
		P1(n, outmask_, layerNr_, layerZ_), P2(n, outmask_, layerNr_, layerZ_), Gen(n, outmask_ & ~(SWEEP_ENH), layerNr_, layerZ_),
		wlsGen(n), betasGen(n)
	{
		outmask = outmask_;
	}

	void SweepResultSecondOrderNLTMM::SetValues(int nr, SecondOrderNLTMM & tmm) {
		wlsGen(nr) = tmm.GetGen()->GetWl();
		betasGen(nr) = tmm.GetGen()->GetBeta();
		
		if (outmask & SWEEP_P1) {
			P1.SetValues(nr, *tmm.GetP1());
		}

		if (outmask & SWEEP_P2) {
			P2.SetValues(nr, *tmm.GetP2());
		}

		if (outmask & SWEEP_GEN) {
			Gen.SetValues(nr, *tmm.GetGen());
		}
	}

	void SecondOrderNLTMM::CheckPrerequisites(TMMParam toIgnore) {
		// Sub TMMs
		tmmP1.CheckPrerequisites(toIgnore);
		tmmP2.CheckPrerequisites(toIgnore);
		tmmGen.CheckPrerequisites(toIgnore);
	}

	void SecondOrderNLTMM::UpdateGenParams()
	{
		double wlP1 = tmmP1.GetWl();
		double wlP2 = tmmP2.GetWl();
		double betaP1 = tmmP1.GetBeta();
		double betaP2 = tmmP2.GetBeta();

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
		case TMM::SPDC:
			wlGen = OmegaToWl(WlToOmega(wlP1) - WlToOmega(wlP2));
			betaGen = wlGen * (betaP1 / wlP1 - betaP2 / wlP2);
			tmmP2.UpdateSPDCParams(deltaWlSpdc, solidAngleSpdc, deltaThetaSpdc, tmmP1.GetWl(), tmmP1.GetBeta());
			break;
		default:
			throw std::runtime_error("Unknown process.");
			break;
		}

		if (wlGen <= 0.0) {
			std::cerr << "Wavelength cannot be negative." << std::endl;
			throw std::runtime_error("Wavelength cannot be negative.");
		}

		tmmGen.SetWl(wlGen);
		tmmGen.SetBeta(betaGen);
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
		case TMM::SPDC:
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

		// Check inhomogenous wave directionality, correct if necessary
		/*
		dcomplex epsL = sqr(material->GetN(tmmGen.GetDouble(PARAM_WL)));
		Polarization pol = (Polarization)tmmGen.GetInt(PARAM_POL);
		WaveDirection dS = F, dA = F;
		if (kpS->kSzF != 0.0) {
			dS = GetWaveDirection(kpS->kSzF, epsL, pol);
		}
		if (kpA->kSzF != 0.0) {
			dA = GetWaveDirection(kpA->kSzF, epsL, pol);
		}

		
		if (dS != F) {
			kpS->kSzF *= -1.0;
			swap(kpS->pF, kpS->pB);
		}

		if (dA != F) {
			kpA->kSzF *= -1.0;
			swap(kpA->pF, kpA->pB);
		}
		*/
		

	}

	void SecondOrderNLTMM::SolveFundamentalFields() {
		// Solve pump 1
		tmmP1.Solve();
		tmmP2.Solve();
	}
	void SecondOrderNLTMM::SolveGeneratedField() {		
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
	
	void SecondOrderNLTMM::SolveWaves(ArrayXd * betasP1, ArrayXcd * E0sP1, ArrayXd * betasP2, ArrayXcd * E0sP2) {
		Material *matLayer0 = tmmP1.GetLayer(0)->GetMaterial();
		Material *matLayerThis = NULL;

		// Solve wave P1
		Wave *waveP1 = tmmP1.GetWave();
		waveP1->Solve(tmmP1.GetWl(), tmmP1.GetBeta(), matLayer0, matLayerThis);
		*betasP1 = waveP1->GetBetas();
		*E0sP1 = waveP1->GetExpansionCoefsKx();

		// Solve wave P2
		Wave *waveP2 = tmmP2.GetWave();
		if (waveP2->GetWaveType() == SPDCWAVE && process != SPDC) {
			std::cerr << "SecondOrderNLTMM must be in SPDC mode to use SPDC wave." << std::endl;
			throw std::invalid_argument("SecondOrderNLTMM must be in SPDC mode to use SPDC wave.");
		}
		double deltaKxSpdc = tmmP2.CalcDeltaKxSpdc();
		waveP2->Solve(tmmP2.GetWl(), tmmP2.GetBeta(), matLayer0, matLayerThis, deltaKxSpdc);
		*betasP2 = waveP2->GetBetas();
		*E0sP2 = waveP2->GetExpansionCoefsKx();
	}
	SecondOrderNLTMM::SecondOrderNLTMM()
	{
		SetProcess(SFG);
		tmmP1.SetMode(MODE_INCIDENT);
		tmmP2.SetMode(MODE_INCIDENT);
		tmmGen.SetMode(MODE_NONLINEAR);
		deltaWlSpdc = constNAN;
		solidAngleSpdc = constNAN;
		deltaThetaSpdc = constNAN;
	}

	void SecondOrderNLTMM::SetProcess(NonlinearProcess process_)
	{
		process = process_;
		if (process == SPDC) {
			tmmP2.SetMode(MODE_VACUUM_FLUCTUATIONS);
		} else {
			tmmP2.SetMode(MODE_INCIDENT);
		}
	}

	void SecondOrderNLTMM::SetDeltaWlSpdc(double value) {
		deltaWlSpdc = value;
	}

	void SecondOrderNLTMM::SetSolidAngleSpdc(double value) {
		solidAngleSpdc = value;
	}

	void SecondOrderNLTMM::SetDeltaThetaSpdc(double value) {
		deltaThetaSpdc = value;
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
		CheckPrerequisites();
		SolveFundamentalFields();
		SolveGeneratedField();
	}

	SecondOrderNLIntensities SecondOrderNLTMM::GetIntensities() const{
		SecondOrderNLIntensities res;
		res.P1 = tmmP1.GetIntensities();
		res.P2 = tmmP2.GetIntensities();
		res.Gen = tmmGen.GetIntensities();
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

	double SecondOrderNLTMM::GetDeltaWlSpdc() {
		return deltaWlSpdc;
	}

	double SecondOrderNLTMM::GetSolidAngleSpdc() {
		return solidAngleSpdc;
	}

	double SecondOrderNLTMM::GetDeltaThetaSpdc() {
		return deltaThetaSpdc;
	}

	SweepResultSecondOrderNLTMM * SecondOrderNLTMM::Sweep(TMMParam param, const Eigen::Map<ArrayXd>& valuesP1, const Eigen::Map<ArrayXd>& valuesP2, int outmask, int paramLayer, int layerNr, double layerZ) {
		if (valuesP1.size() != valuesP2.size()) {
			throw std::invalid_argument("Value arrays must have the same size.");
		}

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
				nlTMMCopy.GetP1()->SetParam(param, valuesP1(i), paramLayer);
				nlTMMCopy.GetP2()->SetParam(param, valuesP2(i), paramLayer);
				if (GetParamType(param) == PTYPE_NONLINEAR_LAYER) {
					// TODO
					nlTMMCopy.GetGen()->SetParam(param, valuesP2(i), paramLayer);
				}

				// Solver
				nlTMMCopy.Solve();
				res->SetValues(i, nlTMMCopy);
			}
		}
		return res;
	}

	FieldsZX * SecondOrderNLTMM::WaveGetFields2D(const Eigen::Map<ArrayXd> &zs, const Eigen::Map<ArrayXd> &xs, WaveDirection dir) {
		// Do checking
		UpdateGenParams();
		CheckPrerequisites();

		// Solve waves
		ArrayXd betasP1, betasP2;
		ArrayXcd E0sP1, E0sP2;
		bool coherent;
		SolveWaves(&betasP1, &E0sP1, &betasP2, &E0sP2);

		// kxs
		ArrayXd kxsP1 = betasP1 * 2.0 * PI / tmmP1.GetWl();
		ArrayXd kxsP2 = betasP2 * 2.0 * PI / tmmP2.GetWl();

		// Allocate space (deletion is the responsibility of the caller!)
		FieldsZX *res = new FieldsZX(zs.size(), xs.size(), tmmGen.GetPolarization());
		res->SetZero();

		if (tmmP1.GetWave()->IsCoherent() && tmmP2.GetWave()->IsCoherent()) {
			#pragma omp parallel
			{
				SecondOrderNLTMM tmmThread = *this;
				tmmThread.tmmP1.SetOverrideE0(true);
				tmmThread.tmmP2.SetOverrideE0(true);
				#pragma omp for
				for (int i = 0; i < betasP1.size(); i++) {
					tmmThread.tmmP1.SetBeta(betasP1(i));
					tmmThread.tmmP1.SetE0(E0sP1(i));
					double dkxP1 = GetDifferential(kxsP1, i);
					for (int j = 0; j < betasP2.size(); j++) {
						tmmThread.tmmP2.SetBeta(betasP2(j));
						tmmThread.tmmP2.SetE0(E0sP2(j));
						double dkxP2 = GetDifferential(kxsP2, j);

						// Solve
						tmmThread.Solve();

						// Integrate fields
						double kxGen = tmmThread.tmmGen.GetLayer(0)->GetKx();
						FieldsZ *fGen = tmmThread.tmmGen.GetFields(zs, dir);
						ArrayXcd phaseXGen = (constI * kxGen * xs).exp() * dkxP1 * dkxP2;
						res->AddFields(*fGen, phaseXGen);
						delete fGen;
					}
				}
			}
		} else if (tmmP1.GetWave()->IsCoherent() && !tmmP2.GetWave()->IsCoherent()){
			#pragma omp parallel
			{
				SecondOrderNLTMM tmmThread = *this;
				tmmThread.tmmP1.SetOverrideE0(true);
				tmmThread.tmmP2.SetOverrideE0(true);
				// Sum field incoherently
				#pragma omp for
				for (int j = 0; j < betasP2.size(); j++) {
					tmmThread.tmmP2.SetBeta(betasP2(j));
					tmmThread.tmmP2.SetE0(E0sP2(j));
					double dkxP2 = GetDifferential(kxsP2, j);

					FieldsZX *coherentFields = new FieldsZX(zs.size(), xs.size(), tmmGen.GetPolarization());
					coherentFields->SetZero();
					// Sum field coherently
					for (int i = 0; i < betasP1.size(); i++) {
						tmmThread.tmmP1.SetBeta(betasP1(i));
						tmmThread.tmmP1.SetE0(E0sP1(i));
						double dkxP1 = GetDifferential(kxsP1, i);
					
						// Solve
						tmmThread.Solve();

						// Integrate fields
						double kxGen = tmmThread.tmmGen.GetLayer(0)->GetKx();
						FieldsZ *fGen = tmmThread.tmmGen.GetFields(zs, dir);
						ArrayXcd phaseXGen = (constI * kxGen * xs).exp() * dkxP1 * dkxP2;
						coherentFields->AddFields(*fGen, phaseXGen);
						delete fGen;
					}

					res->AddSquaredFields(coherentFields);
					delete coherentFields;
				}
			}
			res->TakeSqrt();
		}
		else {
			std::cerr << "Only P2 can be incoherent." << std::endl;
			throw std::invalid_argument("Only P2 can be incoherent.");
		}
	
		return res;
	}

	pairdd SecondOrderNLTMM::WaveGetPowerFlows(int layerNr, double x0, double x1, double z) {
		// Do checking
		if (layerNr < 0 || layerNr > tmmP1.LayersCount()) {
			std::cerr << "Invalid layer index." << std::endl;
			throw std::invalid_argument("Invalid layer index.");
		}
		UpdateGenParams();
		CheckPrerequisites();

		//Solve waves
		ArrayXd betasP1, betasP2;
		ArrayXcd E0sP1, E0sP2;
		SolveWaves(&betasP1, &E0sP1, &betasP2, &E0sP2);
		Wave *waveP1 = tmmP1.GetWave();
		Wave *waveP2 = tmmP2.GetWave();
		double LyP1 = tmmP1.GetWave()->GetLy();
		double LyP2 = waveP2->GetLy();
		
		// TODO
		if (waveP1->GetW0() > waveP2->GetW0()) {
			std::cerr << "Currently P2 must be at least as wide as P1." << std::endl;
			throw std::invalid_argument("Currently P2 must be at least as wide as P1.");
		}

		// X-range (TODO)
		pairdd xrangeP1 = waveP1->GetXRange();
		if (isnan(x0)) {
			x0 = xrangeP1.first;
		}
		if (isnan(x1)) {
			x1 = xrangeP1.second;
		}

		// Ly
		if (LyP1 != LyP2) {
			std::cerr << "Both pump waves must have the same Ly." << std::endl;
			throw std::invalid_argument("Both pump waves must have the same Ly.");
		}

		// Init memory
		int nTot = betasP1.size() * betasP2.size();
		Eigen::MatrixX2cd UsUnsorted(nTot, 2);
		Eigen::MatrixX2cd kzsUnsorted(nTot, 2);
		ArrayXd kxsUnsorted(nTot);

		// Solve for every beta
		bool oldOverrideE0P1 = tmmP1.GetOverrideE0();
		bool oldOverrideE0P2 = tmmP2.GetOverrideE0();
		double oldBetaP1 = tmmP1.GetBeta();
		double oldBetaP2 = tmmP2.GetBeta();
		tmmP1.SetOverrideE0(true);
		tmmP2.SetOverrideE0(true);
		for (int i = 0; i < betasP1.size(); i++) {
			tmmP1.SetBeta(betasP1(i));
			tmmP1.SetE0(E0sP1(i));
			for (int j = 0; j < betasP2.size(); j++) {
				tmmP2.SetBeta(betasP2(j));
				tmmP2.SetE0(E0sP2(j));
				Solve();

				int index = i * betasP2.size() + j;
				UsUnsorted.row(index) = tmmGen.GetLayer(layerNr)->GetMainFields(0.0);
				kzsUnsorted.row(index) = tmmGen.GetLayer(layerNr)->GetHw()->GetKz();
				kxsUnsorted(index) = tmmGen.GetLayer(layerNr)->GetKx();
			}
		}
		tmmP1.SetOverrideE0(oldOverrideE0P1);
		tmmP2.SetOverrideE0(oldOverrideE0P2);
		tmmP1.SetBeta(oldBetaP1);
		tmmP2.SetBeta(oldBetaP2);


		double Ly = LyP1;
		double wlGen = tmmGen.GetWl();
		Polarization polGen = tmmGen.GetPolarization();
		dcomplex epsLayer = tmmGen.GetLayer(layerNr)->eps;
		
		pairdd res(0.0, 0.0);
		if (waveP1->IsCoherent() && waveP2->IsCoherent()) {
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
			ArrayXd kxs(nTot);

			for (int i = 0; i < nTot; i++) {
				Us.row(i) = UsUnsorted.row(indices(i));
				kzs.row(i) = kzsUnsorted.row(indices(i));
				kxs(i) = kxsUnsorted(indices(i));
			}

			// Integrate powers
			res = IntegrateWavePower(layerNr, polGen, wlGen, epsLayer, Us, kxs, kzs, x0, x1, z, Ly);

		} else if (waveP1->IsCoherent() && !waveP2->IsCoherent()) {
			ArrayXd kxsP2 = betasP2 * (2.0 * PI / tmmP2.GetWl());
			for (int j = 0; j < betasP2.size(); j++) {
				// Init
				Eigen::MatrixX2cd Us(betasP1.size(), 2);
				Eigen::MatrixX2cd kzs(betasP1.size(), 2);
				ArrayXd kxs(betasP1.size());

				// Collect amplitutes
				for (int i = 0; i < betasP1.size(); i++) {
					int index = i * betasP2.size() + j;
					Us.row(i) = UsUnsorted.row(index);
					kzs.row(i) = kzsUnsorted.row(index);
					kxs.row(i) = kxsUnsorted.row(index);
				}

				// Sum powers
				double dkxP2 = GetDifferential(kxsP2, j);
				pairdd r = IntegrateWavePower(layerNr, polGen, wlGen, epsLayer, Us, kxs, kzs, x0, x1, z, Ly);
				res.first += r.first * dkxP2;
				res.second += r.second * dkxP2;
			}
		}
		else {
			std::cerr << "Only P2 is allowed to be incoherent." << std::endl;
			throw std::invalid_argument("Only P2 is allowed to be incoherent.");
		}

		
		return res;
	}

	WaveSweepResultSecondOrderNLTMM * SecondOrderNLTMM::WaveSweep(TMMParam param, const Eigen::Map<ArrayXd>& valuesP1, const Eigen::Map<ArrayXd>& valuesP2, int outmask, int paramLayer, int layerNr, double layerZ) {
		if (valuesP1.size() != valuesP2.size()) {
			throw std::invalid_argument("Value arrays must have the same size.");
		}

		CheckPrerequisites(param);

		// Alloc memory for result (dealloc is responsibility of the user)
		WaveSweepResultSecondOrderNLTMM *res = new WaveSweepResultSecondOrderNLTMM(valuesP1.size(), outmask, layerNr, layerZ);

		#pragma omp parallel
		{
			// Make a copy
			SecondOrderNLTMM nlTMMCopy = *this;

			// Sweep
			#pragma omp for
			for (int i = 0; i < valuesP1.size(); i++) {
				//Set sweep param
				nlTMMCopy.GetP1()->SetParam(param, valuesP1(i), paramLayer);
				nlTMMCopy.GetP2()->SetParam(param, valuesP2(i), paramLayer);
				if (GetParamType(param) == PTYPE_NONLINEAR_LAYER) {
					// TODO
					nlTMMCopy.GetGen()->SetParam(param, valuesP2(i), paramLayer);
				}
				nlTMMCopy.UpdateGenParams();
				res->SetValues(i, nlTMMCopy);
			}
		}
		return res;
	}

	WaveSweepResultSecondOrderNLTMM::WaveSweepResultSecondOrderNLTMM(int n, int outmask_, int layerNr_, double layerZ_) :
		P1(n, outmask_, layerNr_, layerZ_), P2(n, outmask_, layerNr_, layerZ_), Gen(n, outmask_ & ~(SWEEP_ENH), layerNr_, layerZ_),
		wlsGen(n), betasGen(n)
	{
		outmask = outmask_;
	}

	void WaveSweepResultSecondOrderNLTMM::SetValues(int nr, SecondOrderNLTMM & tmm) {
		wlsGen(nr) = tmm.GetGen()->GetWl();
		betasGen(nr) = tmm.GetGen()->GetBeta();

		if (outmask & SWEEP_P1) {
			P1.SetValues(nr, *tmm.GetP1());
		}

		if (outmask & SWEEP_P2) {
			P2.SetValues(nr, *tmm.GetP2());
		}

		if (outmask & SWEEP_GEN) {
			// First layer
			if ((outmask & SWEEP_I) || (outmask & SWEEP_R)) {
				pairdd pf0 = tmm.WaveGetPowerFlows(0);
				Gen.Pi(nr) = pf0.first;
				Gen.Pr(nr) = pf0.second;
			}

			// Last layer
			if (outmask & SWEEP_T) {
				pairdd pfL = tmm.WaveGetPowerFlows(tmm.GetGen()->LayersCount() - 1);
				Gen.Pt(nr) = pfL.first;
			}

			Gen.beamArea(nr) = tmm.GetP1()->GetWave()->GetBeamArea();
		}
	}

}
