#pragma once
#include "Common.h"
#include "Material.h"
#include "Waves.h"
#include "NonlinearLayer.h"

namespace TMM {	
	class NonlinearTMM;

	//---------------------------------------------------------------
	// Functions
	//---------------------------------------------------------------

	typedef std::function<void(const ArrayXcd&, const ArrayXcd&, MatrixXcd&)> OuterProductSSEEigenFunc;
	void OuterProductSSEEigenComplex(const ArrayXcd& X, const ArrayXcd& Y, MatrixXcd& R);
	void OuterProductSSEEigenComplexAdd(const ArrayXcd& X, const ArrayXcd& Y, MatrixXcd& R);
	void OuterProductGoodEigenComplex(const ArrayXcd& X, const ArrayXcd& Y, MatrixXcd& R);
	void OuterProductGoodEigenComplexAdd(const ArrayXcd& X, const ArrayXcd& Y, MatrixXcd& R);
	void ThreadSafeMatrixAddNorm(MatrixXcd &mat, const MatrixXcd &toAdd);

	//---------------------------------------------------------------
	// Intensities
	//---------------------------------------------------------------

	class Intensities {
	private:
	public:
		dcomplex inc, r, t;
		double I, R, T;
		
		Intensities(dcomplex inc_ = 0.0, dcomplex r_ = 0.0, dcomplex t_ = 0.0, 
			double I_ = 0.0, double R_ = 0.0, double T_ = 0.0);

		friend std::ostream& operator<<(std::ostream& os, const Intensities& dt);
	};

	//---------------------------------------------------------------
	// SweepResultNonlinearTMM
	//---------------------------------------------------------------

	class SweepResultNonlinearTMM {
	private:
		int outmask;
		int layerNr;
		double layerZ;
	public:
		ArrayXcd inc, r, t;
		ArrayXd Ii, Ir, It, Ia;
		ArrayXd enh;

		SweepResultNonlinearTMM(int n, int outmask_, int layerNr_, double layerZ_);
		int GetOutmask();
		void SetValues(int nr, NonlinearTMM &tmm);
	};

	//---------------------------------------------------------------
	// WaveSweepResultNonlinearTMM
	//---------------------------------------------------------------

	class WaveSweepResultNonlinearTMM {
	private:
		int outmask;
		int layerNr;
		double layerZ;
	public:
		ArrayXd Pi, Pr, Pt;
		ArrayXd enh, beamArea;

		WaveSweepResultNonlinearTMM(int n, int outmask_, int layerNr_, double layerZ_);
		int GetOutmask();
		void SetValues(int nr, NonlinearTMM &tmm);
	};

	//---------------------------------------------------------------
	// FieldsZ
	//---------------------------------------------------------------

	class FieldsZ {
	private:
	public:
		MatrixXcd E, H;
		FieldsZ(int n);
		void SetFieldsAtZ(int nr, const Fields &f);

	};

	//---------------------------------------------------------------
	// FieldsZX
	//---------------------------------------------------------------

	class FieldsZX {
	private:
		Polarization pol;
	public:
		MatrixXcd Ex, Ey, Ez, Hx, Hy, Hz;
		FieldsZX(int n, int m, Polarization pol_);
		void SetZero();
		Polarization GetPol();
		void SetFields(const FieldsZ &f, const ArrayXcd &phaseX, bool add = false);
		void AddFields(const FieldsZ &f, const ArrayXcd &phaseX);
		void AddSquaredFields(FieldsZX *toAdd);
		void TakeSqrt();
		MatrixXd GetENorm();

	};

	//---------------------------------------------------------------
	// NonlinearTMM
	//---------------------------------------------------------------

	class NonlinearTMM {

	private:
		double wl;
		double beta;
		Polarization pol;
		double I0;
		bool overrideE0;
		dcomplex E0;
		NonlinearTmmMode mode;
		std::vector<NonlinearLayer> layers;
		std::vector<Matrix2cd> transferMatrices;
		std::vector<Array2cd> transferMatricesNL;
		std::vector<Matrix2cd> systemMatrices;
		std::vector<Array2cd> systemMatricesNL;
		dcomplex inc, r, t;
		bool solved;
		Wave wave;
		double deltaWlSpdc, solidAngleSpdc, deltaThetaSpdc, wlP1Spdc, betaP1Spdc;

		Array2cd CalcTransferMatrixNL(int interfaceNr, const InhomogeneousWave &w1, const InhomogeneousWave &w2);
		void SolveInterfaceTransferMatrix(int interfaceNr);
		void SolveAllTransferMatrices();
		void SolveSystemMatrix();
		void SolveIncReflTrans();
		void SolveWave(ArrayXd *betas, ArrayXcd *E0s);
		double CalcVacFuctuationsE0();

	public:
		NonlinearTMM();
		void AddLayer(double d_, Material *material_);
		NonlinearLayer* GetLayer(int layerNr);
		int LayersCount() const;
		void CheckPrerequisites(TMMParam toIgnore = PARAM_NOT_DEFINED);

		// Setters
		void SetWl(double wl_);
		void SetBeta(double beta_);
		void SetPolarization(Polarization pol_);
		void SetI0(double I0_);
		void SetOverrideE0(bool overrideE0_);
		void SetE0(dcomplex E0_);
		void SetMode(NonlinearTmmMode mode_);
		void SetParam(TMMParam param, double value, int paramLayer = -1);
		void SetParam(TMMParam param, dcomplex value, int paramLayer = -1);

		// Getters
		double GetWl();
		double GetBeta();
		Polarization GetPolarization();
		double GetI0();
		bool GetOverrideE0();
		dcomplex GetE0();
		NonlinearTmmMode GetMode();
		double GetDouble(TMMParam param);
		dcomplex GetComplex(TMMParam param);

		// Plane wave functionality
		void Solve();
		Intensities GetIntensities() const;
		double GetAbsorbedIntensity() const;
		double GetEnhancement(int layerNr, double z);
		SweepResultNonlinearTMM* Sweep(TMMParam param, const Eigen::Map<ArrayXd> &values, int outmask = SWEEP_PWRFLOWS, int paramLayer = -1, int layerNr = 0, double layerZ = 0);
		FieldsZ* GetFields(const Eigen::Map<ArrayXd> &zs, WaveDirection dir = TOT);
		FieldsZX* GetFields2D(const Eigen::Map<ArrayXd> &zs, const Eigen::Map<ArrayXd> &xs, WaveDirection dir = TOT);

		// Waves
		Wave* GetWave();
		pairdd WaveGetPowerFlows(int layerNr, double x0 = constNAN, double x1 = constNAN, double z = 0.0);
		double WaveGetEnhancement(int layerNr, double z);
		WaveSweepResultNonlinearTMM * WaveSweep(TMMParam param, const Eigen::Map<ArrayXd> &values, int outmask = SWEEP_PWRFLOWS, int paramLayer = -1, int layerNr = 0, double layerZ = 0);
		FieldsZX* WaveGetFields2D(const Eigen::Map<ArrayXd> &zs, const Eigen::Map<ArrayXd> &xs, WaveDirection dir = TOT);

		// SPDC (used internally by SecondOrderNLTMM)
		void UpdateSPDCParams(double deltaWlSpdc_, double solidAngleSpdc_, double deltaThetaSpdc_, double wlP1Spdc_, double betaP1Spdc_);
		double CalcDeltaKxSpdc();
	};

}
