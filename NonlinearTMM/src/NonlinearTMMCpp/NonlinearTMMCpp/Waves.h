#pragma once
#include "Common.h"
#include "Material.h"
#include <unsupported/Eigen/FFT>

namespace TMM {

	//---------------------------------------------------------------
	// ENUMs
	//---------------------------------------------------------------

	enum class WaveType {
		PLANEWAVE,
		GAUSSIANWAVE,
		TUKEYWAVE,
		SPDCWAVE,
	};

	//---------------------------------------------------------------
	// Functions
	//---------------------------------------------------------------

	ArrayXd TukeyFunc(const ArrayXd& xs, double w0, double a);
	
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
		WaveType GetWaveType() const;
		double GetPwr() const;
		bool GetOverrideE0() const;
		double GetE0() const;
		double GetW0() const;
		double GetLy() const;
		double GetA() const;
		int GetNPointsInteg() const;
		double GetMaxX() const;
		bool IsDynamicMaxXEnabled() const;
		double GetDynamicMaxXCoef() const;
		double GetDynamicMaxXAddition() const;
		double GetMaxPhi() const;
		double GetDouble(TMMParam param) const;
		pairdd GetXRange() const;
		ArrayXd GetBetas() const;
		ArrayXd GetPhis() const;
		ArrayXd GetKxs() const;
		ArrayXd GetKzs() const;
		ArrayXd GetFieldProfileXs() const;
		ArrayXd GetFieldProfile() const;
		ArrayXcd GetExpansionCoefsKx() const;
		double GetBeamArea() const;
		bool IsCoherent() const;
	};


}
