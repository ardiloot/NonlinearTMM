#pragma once
#include "Common.h"
#include "Material.h"

namespace TMM {
	//---------------------------------------------------------------
	// Forward declarations
	//---------------------------------------------------------------

	class NonlinearLayer;

	//---------------------------------------------------------------
	// Fields
	//---------------------------------------------------------------

	class Fields {
	private:
		WaveDirection dir;
		
	public:
		Vector3cd E;
		Vector3cd H;

		Fields();
		void SetFieldsPpol(const Array2cd &Hy, const Array2cd &Ex, const Array2cd &Ez, WaveDirection dir_);
		void SetFieldsSpol(const Array2cd &Ey, const Array2cd &Hx, const Array2cd &Hz, WaveDirection dir_);
	};

	//---------------------------------------------------------------
	// InhomogeneosWaveParams
	//---------------------------------------------------------------

	class InhomogeneosWaveParams {
	private:

	public:
		dcomplex kSzF;
		Vector3cd pF;
		Vector3cd pB;
		InhomogeneosWaveParams();
	};

	//---------------------------------------------------------------
	// HomogeneousWave
	//---------------------------------------------------------------

	class HomogeneousWave {
		friend class InhomogeneousWave;
		friend class NonlinearLayer;
		friend class NonlinearTMM;
	private:
		NonlinearLayer *layer; //this pointer is wrong after copy, updated after each solve call
		Array2cd kz;
		Array2cd phase; // complex phase from the beginning to the end
		Matrix2cd propMatrix;
		bool solved;

		void Solve(NonlinearLayer *layer_);

	public:
		HomogeneousWave(NonlinearLayer *layer_ = NULL);
		double GetKx() const;
		Array2cd GetKz() const;
		dcomplex GetKzF() const;
		Array2cd GetMainFields(double z) const;
	};

	//---------------------------------------------------------------
	// InhomogeneousWave
	//---------------------------------------------------------------

	class InhomogeneousWave {
		friend class NonlinearLayer;
		friend class NonlinearTMM;
	private:
		NonlinearLayer *layer;
		Array2cd kSz;
		dcomplex kSNorm;
		Array2cd phaseS;
		Array2cd px, py, pz;
		Array2cd By;
		Array2cd propMatrixNL;
		bool solved;

		void Solve(NonlinearLayer *layer_, InhomogeneosWaveParams &params);

	public:
		InhomogeneousWave(NonlinearLayer *layer_ = NULL);
		Array2cd GetMainFields(double z) const;
	};

	//---------------------------------------------------------------
	// NonlinearLayer
	//---------------------------------------------------------------

	class NonlinearLayer {
		friend class HomogeneousWave;
		friend class InhomogeneousWave;
		friend class NonlinearTMM;
		friend class SecondOrderNLTMM;
	private:
		double d;
		Material *material;
		HomogeneousWave hw;
		InhomogeneousWave iws;
		InhomogeneousWave iwa;
		double wl;
		double beta;
		Polarization pol;
		double omega;
		dcomplex n;
		dcomplex eps;
		double k0;
		dcomplex k;
		dcomplex kNorm;
		double kx;
		Matrix2cd propMatrix;
		Array2cd propMatrixNL;
		bool isNonlinear;
		InhomogeneosWaveParams kpS, kpA;
		Array2cd U0;
		bool solved;

		void Solve(double wl_, double beta_, Polarization pol_);
	
	public:
		NonlinearLayer(double d_, Material *material_);
		
		bool IsNonlinear() const;
		void SetNonlinearity(InhomogeneosWaveParams kpS_, InhomogeneosWaveParams kpA_);
		void ClearNonlinearity();
		
		void SetThickness(double d_);
		void SetParam(TMMParam param, double value);

		double GetThickness();
		Material * GetMaterial();
		HomogeneousWave* GetHw();
		double GetKx() const;
		double GetK0() const;
		double GetDouble(TMMParam param);

		Array2cd GetMainFields(double z) const;
		Fields GetFields(double z, WaveDirection dir = TOT) const;
		double GetIntensity(double z) const;
		double GetAbsorbedIntensity() const;
		double GetSrcIntensity() const;
		double GetENorm(double z, WaveDirection dir = TOT) const;
		
	};
}