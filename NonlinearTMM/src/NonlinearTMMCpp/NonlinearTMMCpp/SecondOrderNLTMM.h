#pragma once
#include "Common.h"
#include "NonlinearTMM.h"

namespace TMM {

	//---------------------------------------------------------------
	// SecondOrderNLPowerFlows
	//---------------------------------------------------------------

	class SecondOrderNLPowerFlows {
	private:
	public:
		PowerFlows P1, P2, Gen;
	};

	//---------------------------------------------------------------
	// SweepResultSecondOrderNLTMM
	//---------------------------------------------------------------

	class SweepResultSecondOrderNLTMM {
	private:
	public:
		SweepResultNonlinearTMM P1, P2, Gen;
		SweepResultSecondOrderNLTMM(int n);
		void SetPowerFlows(int nr, const SecondOrderNLPowerFlows &pw);
	};

	//---------------------------------------------------------------
	// SecondOrderNLTMM
	//---------------------------------------------------------------

	class SecondOrderNLTMM {
	private:
		double wlGen, betaGen;
		NonlinearProcess process;
		NonlinearTMM tmmP1, tmmP2, tmmGen;

		void UpdateGenParams();
		void CalcInhomogeneosWaveParams(int layerNr, Material *material, InhomogeneosWaveParams *kpS, InhomogeneosWaveParams *kpA);
		__declspec(noinline) void SolveFundamentalFields();
		__declspec(noinline) void SolveGeneratedField();

	public:
		SecondOrderNLTMM();
		void SetProcess(NonlinearProcess process_);
		void AddLayer(double d_, Material *material_);
		void Solve();
		SecondOrderNLPowerFlows GetPowerFlows();
		NonlinearTMM* GetP1();
		NonlinearTMM* GetP2();
		NonlinearTMM* GetGen();
		SweepResultSecondOrderNLTMM* Sweep(TMMParam param, const Eigen::Map<Eigen::ArrayXd> &valuesP1, const Eigen::Map<Eigen::ArrayXd> &valuesP2);
		FieldsZX * GetGenWaveFields2D(const Eigen::Map<Eigen::ArrayXd>& betasP1, const Eigen::Map<Eigen::ArrayXd>& betasP2, const Eigen::Map<Eigen::ArrayXcd>& E0sP1, const Eigen::Map<Eigen::ArrayXcd>& E0sP2, const Eigen::Map<Eigen::ArrayXd>& zs, const Eigen::Map<Eigen::ArrayXd>& xs, WaveDirection dir = TOT);

	};
}