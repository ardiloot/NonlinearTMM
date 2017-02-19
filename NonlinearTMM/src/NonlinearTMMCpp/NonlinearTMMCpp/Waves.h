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
		Material *material;
		double Ly;
		double a; // Tukey param
		int nPointsInteg;
		double maxPhi;
		double integCriteria;
		double maxX;

		double wl;
		double beta;
		double k0;
		ArrayXd kxs, kzs, fieldProfileXs, fieldProfile;
		ArrayXcd expansionCoefsKx;
		Eigen::FFT<double> fft;

		void SolvePlaneWave(double E0, double th0, double k);
		void SolveFFTWave(double E0, double th0, double k, int iteration, double maxPhiThis);

	public:
	
		Wave();

		// Setters
		void SetWaveType(WaveType waveType_);
		void SetPwr(double pwr_);
		void SetOverrideE0(bool overrideE0_);
		void SetE0(double E0_);
		void SetW0(double w0_);
		void SetMaterial(Material *material_);
		void SetLy(double Ly_);
		void SetA(double a_);
		void SetNPointsInteg(int nPointsInteg_);
		void SetMaxPhi(double maxPhi_);
		void SetIntegCriteria(double criteria_);
		void SetMaxX(double maxX_);

		// Solve
		void Solve(double wl, double beta, Material *material_ = NULL);

		// Getters
		double GetPwr();
		bool GetOverrideE0();
		double GetE0();
		double GetW0();
		double GetLy();
		double GetA();
		int GetNPointsInteg();
		double GetMaxPhi();
		double GetIntegCriteria();
		double GetMaxX();
		pairdd GetXRange();
		ArrayXd GetBetas();
		ArrayXd GetKxs();
		ArrayXd GetKzs();
		ArrayXd GetFieldProfileXs();
		ArrayXd GetFieldProfile();
		ArrayXcd GetExpansionCoefsKx();
	};


}
