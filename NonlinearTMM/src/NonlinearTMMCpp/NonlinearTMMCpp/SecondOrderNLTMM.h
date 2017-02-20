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
		ArrayXd wlsGen, betasGen;
		SweepResultSecondOrderNLTMM(int n, int outmask, int layerNr_, double layerZ_);
		void SetValues(int nr, SecondOrderNLTMM &tmm);
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
		void SolveFundamentalFields();
		void SolveGeneratedField();

	public:
		SecondOrderNLTMM();
		void SetProcess(NonlinearProcess process_);
		void AddLayer(double d_, Material *material_);
		NonlinearTMM* GetP1();
		NonlinearTMM* GetP2();
		NonlinearTMM* GetGen();
		
		// Planewaves
		void Solve();
		SecondOrderNLPowerFlows GetPowerFlows() const;
		SweepResultSecondOrderNLTMM* Sweep(TMMParam param, const Eigen::Map<ArrayXd> &valuesP1, const Eigen::Map<ArrayXd> &valuesP2, int outmask = 1, int paramLayer = -1, int layerNr = 0, double layerZ = 0.0);

		// Waves
		pairdd WaveGetPowerFlows(int layerNr, double x0 = constNAN, double x1 = constNAN, double z = 0.0);
		FieldsZX * WaveGetFields2D(const Eigen::Map<ArrayXd> &zs, const Eigen::Map<ArrayXd> &xs, WaveDirection dir = TOT);
		
	};
}