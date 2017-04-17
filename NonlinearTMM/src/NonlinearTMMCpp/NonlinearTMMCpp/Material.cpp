#include "Material.h"

namespace TMM {

	Chi2Tensor::Chi2Tensor() : chi2(3, 3, 3), chi2Rotated(3, 3, 3) {
		Clear();
		SetDistinctFields(true);
		SetRotation(0.0, 0.0, 0.0);
	}

	void Chi2Tensor::SetDistinctFields(bool distinctFields_) {
		distinctFields = distinctFields_;
	}

	void Chi2Tensor::SetRotation(double phiX_, double phiY_, double phiZ_) {
		phiX = phiX_;
		phiY = phiY_;
		phiZ = phiZ_;
		needRotationRecalc = true;
	}

	void Chi2Tensor::SetChi2(int i1, int i2, int i3, double value) {
		// indices are from 1..3
		isNonlinear = true;

		if (std::max<int>(i1, std::max<int>(i2, i3)) > 3 || std::min<int>(i1, std::min<int>(i2, i3)) < 1) {
			throw std::runtime_error("chi2 index not in range 1..3");
		}

		needRotationRecalc = true;
		chi2(i1 - 1, i2 - 1, i3 - 1) = value;
	}

	void Chi2Tensor::SetD(int i1, int i2, double value) {
		// indices are from i1=1..3, i2=1..6
		isNonlinear = true;

		if (std::min<int>(i1, i2) < 1 || i1 > 3 || i2 > 6) {
			throw std::runtime_error("d index not in range 1..3, 1..6");
		}
		needRotationRecalc = true;
		int dIndex2[6] = { 0, 1, 2, 1, 0, 0 };
		int dIndex3[6] = { 0, 1, 2, 2, 2, 1 };
		chi2(i1 - 1, dIndex2[i2 - 1], dIndex3[i2 - 1]) = 2.0 * value;
		chi2(i1 - 1, dIndex3[i2 - 1], dIndex2[i2 - 1]) = 2.0 * value;
	}

	void Chi2Tensor::Clear()
	{
		chi2.setZero();
		chi2Rotated.setZero();
		isNonlinear = false;
		needRotationRecalc = false;
	}

	bool Chi2Tensor::IsNonlinear() const
	{
		return isNonlinear;
	}

	const Tensor3d & Chi2Tensor::GetChi2Tensor() {
		if (needRotationRecalc) {
			chi2Rotated = RotateTensor(chi2, phiX, phiY, phiZ);
			needRotationRecalc = false;
		}
		return chi2Rotated;
	}

	double Chi2Tensor::GetChi2Element(int i1, int i2, int i3) {
		// Indices form 1..3
		if (std::max<int>(i1, std::max<int>(i2, i3)) > 3 || std::min<int>(i1, std::min<int>(i2, i3)) < 1) {
			throw std::invalid_argument("Indices must be in range 1..3");
		}

		double res = GetChi2Tensor()(i1 - 1, i2 - 1, i3 - 1);
		return res;
	}

	Vector3cd Chi2Tensor::GetNonlinearPolarization(const Vector3cd &E1, const Vector3cd &E2) {
		Vector3cd res = Vector3cd::Zero();
		Tensor3d chi2 = GetChi2Tensor();

		for (int i = 0; i < 3; i++) {
			res(i) = E1(0) * (E2(0) * chi2(i, 0, 0) + E2(1) * chi2(i, 0, 1) + E2(2) * chi2(i, 0, 2)) +
				E1(1) * (E2(0) * chi2(i, 1, 0) + E2(1) * chi2(i, 1, 1) + E2(2) * chi2(i, 1, 2)) +
				E1(2) * (E2(0) * chi2(i, 2, 0) + E2(1) * chi2(i, 2, 1) + E2(2) * chi2(i, 2, 2));
		}
		res *= constEps0;
		if (distinctFields) {
			res *= 2.0;
		}
		return res;
	}

	Material::Material(dcomplex n_) {
		isStatic = true;
		staticN = n_;
	}

	Material::Material(ArrayXd wlsExp_, ArrayXcd nsExp_) : wlsExp(wlsExp_), nsExp(nsExp_) {
		// Copy of wlsExp and nsExp is made intentionally
		isStatic = false;
		staticN = 1.0;

		if (wlsExp.size() != nsExp.size()) {
			throw std::invalid_argument("wls and ns must have the same length");
		}

		if (wlsExp.size() < 2) {
			throw std::invalid_argument("The length of wls and ns must be at least 2");
		}
	}

	dcomplex Material::GetN(double wl) const {
		if (isStatic) {
			return staticN;
		}

		// Interpolate
		dcomplex res = Interpolate(wl, wlsExp, nsExp);
		return res;
	}
	
	bool Material::IsNonlinear() const
	{
		return chi2.IsNonlinear();
	}
	
}