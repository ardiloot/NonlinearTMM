#pragma once
#include "Common.h"

namespace TMM {

	//---------------------------------------------------------------
	// Chi2Tensor
	//---------------------------------------------------------------

	class Chi2Tensor {
	private:
		Tensor3d chi2;
		Tensor3d chi2Rotated;
		bool needRotationRecalc;
		bool distinctFields;
		bool isNonlinear;
		double phiX, phiY, phiZ;
	public:
		Chi2Tensor();
		void SetDistinctFields(bool distinctFields_);
		void SetRotation(double phiX_, double phiY_, double phiZ_);
		void SetChi2(int i1, int i2, int i3, double value);
		void SetD(int i1, int i2, double value);
		void Clear();

		bool IsNonlinear() const;
		const Tensor3d & GetChi2Tensor();
		double GetChi2Element(int i1, int i2, int i3);
		Vector3cd GetNonlinearPolarization(const Vector3cd &E1, const Vector3cd &E2);
	};

	//---------------------------------------------------------------
	// Material
	//---------------------------------------------------------------

	class Material {
	private:
		bool isStatic;
		dcomplex staticN;
		ArrayXd wlsExp;
		ArrayXcd nsExp;
	public:
		Chi2Tensor chi2;

		Material(dcomplex n_);
		Material(ArrayXd wlsExp_, ArrayXcd nsExp_);
		dcomplex GetN(double wl) const;
		bool IsNonlinear() const;
	};
}
