#pragma once
#include "Common.h"
#include "Material.h"
#include <unsupported/Eigen/FFT>

namespace TMM {

	//---------------------------------------------------------------
	// ENUMs
	//---------------------------------------------------------------

	enum WaveType {
		PLANEWAVE,
		GAUSSIANWAVE,
		TUKEYWAVE,
		SPDCWAVE,
	};

	//---------------------------------------------------------------
	// Functions
	//---------------------------------------------------------------

	ArrayXd TukeyFunc(ArrayXd xs, double w0, double a);
	
	//---------------------------------------------------------------
	// Wave
	//---------------------------------------------------------------

	class Wave {
	private:
		WaveType waveType;
		double pwr;
		bool overrideE0;
		double E0OverrideValue;
		double w0;
		Material *materialLayer0;
		Material *materialLayerThis;
		double Ly;
		double a; // Tukey param
		int nPointsInteg;
		double maxX;
		bool dynamicMaxX;
		double dynamicMaxXCoef;
		double dynamicMaxXAddition;
		double maxXThis;
		double maxPhi;
		double deltaKxSpdc; // SPDC
		bool solved;

		double wl;
		double beta;
		double k0;
		ArrayXd phis, kxs, kzs, fieldProfileXs, fieldProfile;
		ArrayXcd expansionCoefsKx;
		Eigen::FFT<double> fft;
		double E0;
		double k;
		double nLayer0;
		double thLayer0;
		double beamArea;

		void SolvePlaneWave();
		void SolveFFTWave();
		void SolveSpdcWave();

	public:

		Wave();

		// Setters
		void SetWaveType(WaveType waveType_);
		void SetPwr(double pwr_);
		void SetOverrideE0(bool overrideE0_);
		void SetE0(double E0_);
		void SetW0(double w0_);
		void SetLy(double Ly_);
		void SetA(double a_);
		void SetNPointsInteg(int nPointsInteg_);
		void SetMaxX(double maxX_);
		void EnableDynamicMaxX(bool dynamicMaxX_);
		void SetDynamicMaxXCoef(double dynamicMaxXCoef_);
		void SetDynamicMaxXAddition(double dynamicMaxXAddition_);
		void SetMaxPhi(double maxPhi_);
		void SetParam(TMMParam param, double value);

		// Solve
		void Solve(double wl_, double beta_, Material *materialLayer0_, Material *materialLayerThis_, double deltaKxSpdc_ = constNAN);

		// Getters
		WaveType GetWaveType();
		double GetPwr();
		bool GetOverrideE0();
		double GetE0();
		double GetW0();
		double GetLy();
		double GetA();
		int GetNPointsInteg();
		double GetMaxX();
		bool IsDynamicMaxXEnabled();
		double GetDynamicMaxXCoef();
		double GetDynamicMaxXAddition();
		double GetMaxPhi();
		double GetDouble(TMMParam param);
		pairdd GetXRange();
		ArrayXd GetBetas();
		ArrayXd GetPhis();
		ArrayXd GetKxs();
		ArrayXd GetKzs();
		ArrayXd GetFieldProfileXs();
		ArrayXd GetFieldProfile();
		ArrayXcd GetExpansionCoefsKx();
		double GetBeamArea();
		bool IsCoherent();
	};


}
