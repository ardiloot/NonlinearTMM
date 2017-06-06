#pragma once
#include "Common.h"
#include "NonlinearTMM.h"

namespace TMM {

	//---------------------------------------------------------------
	// SecondOrderNLIntensities
	//---------------------------------------------------------------

	class SecondOrderNLIntensities {
	private:
	public:
		Intensities P1, P2, Gen;
	};

	//---------------------------------------------------------------
	// SweepResultSecondOrderNLTMM
	//---------------------------------------------------------------

	class SweepResultSecondOrderNLTMM {
	private:
		int outmask;
	public:
		SweepResultNonlinearTMM P1, P2, Gen;
		ArrayXd wlsGen, betasGen;
		SweepResultSecondOrderNLTMM(int n, int outmask, int layerNr_, double layerZ_);
		void SetValues(int nr, SecondOrderNLTMM &tmm);
	};

	//---------------------------------------------------------------
	// WaveSweepResultSecondOrderNLTMM
	//---------------------------------------------------------------

	class WaveSweepResultSecondOrderNLTMM {
	private:
		int outmask;
	public:
		WaveSweepResultNonlinearTMM P1, P2, Gen;
		ArrayXd wlsGen, betasGen;
		WaveSweepResultSecondOrderNLTMM(int n, int outmask_, int layerNr_, double layerZ_);
		void SetValues(int nr, SecondOrderNLTMM &tmm);
	};

	//---------------------------------------------------------------
	// SecondOrderNLTMM
	//---------------------------------------------------------------

	class SecondOrderNLTMM {
	private:
		double wlGen, betaGen;
		double deltaWlSpdc, solidAngleSpdc, deltaThetaSpdc;
		NonlinearProcess process;
		NonlinearTMM tmmP1, tmmP2, tmmGen;

		void CalcInhomogeneosWaveParams(int layerNr, Material *material, InhomogeneosWaveParams *kpS, InhomogeneosWaveParams *kpA);
		void SolveFundamentalFields();
		void SolveGeneratedField();
		void SolveWaves(ArrayXd *betasP1, ArrayXcd *E0sP1, ArrayXd *betasP2, ArrayXcd *E0sP2);

	public:
		SecondOrderNLTMM();
		void SetProcess(NonlinearProcess process_);
		void SetDeltaWlSpdc(double value);
		void SetSolidAngleSpdc(double value);
		void SetDeltaThetaSpdc(double value);
		void AddLayer(double d_, Material *material_);
		NonlinearTMM* GetP1();
		NonlinearTMM* GetP2();
		NonlinearTMM* GetGen();
		double GetDeltaWlSpdc();
		double GetSolidAngleSpdc();
		double GetDeltaThetaSpdc();

		void CheckPrerequisites(TMMParam toIgnore = PARAM_NOT_DEFINED);
		void UpdateGenParams();
		
		// Planewaves
		void Solve();
		SecondOrderNLIntensities GetIntensities() const;
		SweepResultSecondOrderNLTMM* Sweep(TMMParam param, const Eigen::Map<ArrayXd> &valuesP1, const Eigen::Map<ArrayXd> &valuesP2, int outmask = SWEEP_ALL_WAVE_PWRS, int paramLayer = -1, int layerNr = 0, double layerZ = 0.0);

		// Waves
		pairdd WaveGetPowerFlows(int layerNr, double x0 = constNAN, double x1 = constNAN, double z = 0.0);
		WaveSweepResultSecondOrderNLTMM * WaveSweep(TMMParam param, const Eigen::Map<ArrayXd>& valuesP1, const Eigen::Map<ArrayXd>& valuesP2, int outmask = SWEEP_ALL_WAVE_PWRS, int paramLayer = -1, int layerNr = 0, double layerZ = 0.0);
		FieldsZX * WaveGetFields2D(const Eigen::Map<ArrayXd> &zs, const Eigen::Map<ArrayXd> &xs, WaveDirection dir = TOT);
		
	};
}