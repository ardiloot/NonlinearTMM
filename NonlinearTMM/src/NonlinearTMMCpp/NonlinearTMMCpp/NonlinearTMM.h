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

	// Complex multiplication with SSE
	__forceinline dcomplex multSSE(dcomplex aa, dcomplex bb) {
		const __m128d mask = _mm_set_pd(-0.0, 0.0);

		// Load to registers
		__m128d a = _mm_load_pd((double*)&aa);
		__m128d b = _mm_load_pd((double*)&bb);

		// Real part
		__m128d ab = _mm_mul_pd(a, b);
		ab = _mm_xor_pd(ab, mask);

		// Imaginary part
		b = _mm_shuffle_pd(b, b, 1);
		b = _mm_mul_pd(b, a);

		// Combine
		ab = _mm_hadd_pd(ab, b);

		dcomplex res = 0.0;
		_mm_storeu_pd((double*)&(res), ab);
		return res;
	}

	typedef std::function<void(const ArrayXcd&, const ArrayXcd&, MatrixXcd&)> OuterProductSSEEigenFunc;
	void OuterProductSSEEigenComplex(const ArrayXcd& X, const ArrayXcd& Y, MatrixXcd& R);
	void OuterProductSSEEigenComplexAdd(const ArrayXcd& X, const ArrayXcd& Y, MatrixXcd& R);
	void OuterProductGoodEigenComplex(const ArrayXcd& X, const ArrayXcd& Y, MatrixXcd& R);
	void OuterProductGoodEigenComplexAdd(const ArrayXcd& X, const ArrayXcd& Y, MatrixXcd& R);

	//---------------------------------------------------------------
	// PowerFlows
	//---------------------------------------------------------------

	class PowerFlows {
	private:
	public:
		dcomplex inc, r, t;
		double I, R, T;
		
		PowerFlows(dcomplex inc_ = 0.0, dcomplex r_ = 0.0, dcomplex t_ = 0.0, 
			double I_ = 0.0, double R_ = 0.0, double T_ = 0.0);

		friend std::ostream& operator<<(std::ostream& os, const PowerFlows& dt);
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
		ArrayXd I, R, T, A;
		ArrayXd enh;

		SweepResultNonlinearTMM(int n, int outmask_, int layerNr_, double layerZ_);
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

		Array2cd CalcTransferMatrixNL(int interfaceNr, const InhomogeneousWave &w1, const InhomogeneousWave &w2);
		void SolveInterfaceTransferMatrix(int interfaceNr);
		void SolveAllTransferMatrices();
		void SolveSystemMatrix();
		void SolveIncReflTrans();

	public:
		NonlinearTMM();
		void AddLayer(double d_, Material *material_);
		NonlinearLayer* GetLayer(int layerNr);
		int LayersCount() const;
		void CheckPrerequisites(TMMParam toIgnore = PARAM_NOT_DEFINED);
		
		void Solve();
		PowerFlows GetPowerFlows() const;
		double GetAbsorbedPower() const;
		double GetEnhancement(int layerNr, double z);
		SweepResultNonlinearTMM* Sweep(TMMParam param, const Eigen::Map<ArrayXd> &values, int outmask = 1, int layerNr = 0, double layerZ = 0);
		FieldsZ* GetFields(const Eigen::Map<ArrayXd> &zs, WaveDirection dir = TOT);
		FieldsZX* GetFields2D(const Eigen::Map<ArrayXd> &zs, const Eigen::Map<ArrayXd> &xs, WaveDirection dir = TOT);

		pairdd GetPowerFlowsForWave(const Eigen::Map<ArrayXd> &betas, const Eigen::Map<ArrayXcd> &E0s,
			int layerNr, double x0, double x1, double z, double Ly, WaveDirection dir = TOT);
		FieldsZX* GetWaveFields2D(const Eigen::Map<ArrayXd> &betas, const Eigen::Map<ArrayXcd> &E0s, const Eigen::Map<ArrayXd> &zs, const Eigen::Map<ArrayXd> &xs, WaveDirection dir = TOT);
		
		// Setters
		void SetParam(TMMParam param, bool value);
		void SetParam(TMMParam param, int value);
		void SetParam(TMMParam param, double value);
		void SetParam(TMMParam param, dcomplex value);
		
		// Getters
		bool GetBool(TMMParam param);
		int GetInt(TMMParam param);
		double GetDouble(TMMParam param);
		dcomplex GetComplex(TMMParam param);
	};

}
