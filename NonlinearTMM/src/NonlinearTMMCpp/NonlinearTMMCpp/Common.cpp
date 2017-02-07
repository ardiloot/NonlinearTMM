#include "Common.h"

namespace TMM {
	
	double WlToOmega(double wl) {
		double omega = 2.0 * PI * constC / wl;
		return omega;
	}

	double OmegaToWl(double omega)
	{
		double wl = 2.0 * PI * constC / omega;
		return wl;
	}

	Eigen::Matrix3d RotationMatrixX(double phi) {
		Eigen::Matrix3d res;
		res << 1.0, 0.0, 0.0, 0.0, std::cos(phi), -std::sin(phi), 0.0, std::sin(phi), std::cos(phi);
		return res;
	}

	Eigen::Matrix3d RotationMatrixY(double phi) {
		Eigen::Matrix3d res;
		res << std::cos(phi), 0.0, std::sin(phi), 0.0, 1.0, 0.0, -std::sin(phi), 0.0, std::cos(phi);
		return res;
	}

	Eigen::Matrix3d RotationMatrixZ(double phi) {
		Eigen::Matrix3d res;
		res << std::cos(phi), -std::sin(phi), 0.0, std::sin(phi), std::cos(phi), 0.0, 0.0, 0.0, 1.0;
		return res;
	}

	Tensor3d ApplyRotationMatrixToTensor(Tensor3d input, Eigen::Matrix3d R) {
		Tensor3d res(3, 3, 3);
		res.setZero();

		for (int i1 = 0; i1 < 3; i1++) {
			for (int i2 = 0; i2 < 3; i2++) {
				for (int i3 = 0; i3 < 3; i3++) {
					for (int j1 = 0; j1 < 3; j1++) {
						for (int j2 = 0; j2 < 3; j2++) {
							for (int j3 = 0; j3 < 3; j3++) {
								res(i1, i2, i3) += R(i1, j1) * R(i2, j2) * R(i3, j3) * input(j1, j2, j3);
							}
						}
					}
				}
			}
		}
		return res;
	}

	Tensor3d RotateTensor(Tensor3d & input, double phiX, double phiY, double phiZ)
	{
		Tensor3d output(3, 3, 3);
		output = input;

		Eigen::Matrix3d Rx = RotationMatrixX(phiX);
		Eigen::Matrix3d Ry = RotationMatrixY(phiY);
		Eigen::Matrix3d Rz = RotationMatrixZ(phiZ);

		output = ApplyRotationMatrixToTensor(output, Rx);
		output = ApplyRotationMatrixToTensor(output, Ry);
		output = ApplyRotationMatrixToTensor(output, Rz);
		return output;
	}

	double sqr(double a) {
		return a * a;
	}

	dcomplex sqr(dcomplex a) {
		return a * a;
	}

	template<typename T> T Interpolate(double x, const Eigen::ArrayXd & xs, const Eigen::Array<T, Eigen::Dynamic, 1> & ys) {
		// xs must be sorted

		// Check range
		if (x < xs(0) || x >= xs(xs.size() - 1)) {
			throw std::runtime_error("Interpolation out of range");
		}

		if (xs(0) >= xs(xs.size() - 1)) {
			throw std::runtime_error("Interpolation: xs must be sorted");
		}

		// Binary search (last element, that is less or equal than x)
		int b = 0, e = xs.size() - 1;
		while (b < e) {
			int a = (b + e) / 2;
			if (xs(b) >= xs(e)) {
				throw std::runtime_error("Interpolation: xs must be sorted");
			}

			if (xs(a) > x) {
				// [b..a[
				e = a - 1;

				if (xs(e) <= x) {
					b = e;
				}
			}
			else {
				// [a..e]
				b = a; 
				if (xs(a + 1) > x) {
					e = a;
				}
			}
		}
		// Linear interpolation in range x[b]..x[b+1]
		double dx = xs(b + 1) - xs(b);
		T dy = ys(b + 1) - ys(b);
		T res = ys(b) + dy / dx * (x - xs(b));
		return res;
	}

	template dcomplex Interpolate<dcomplex>(double, const Eigen::ArrayXd &, const Eigen::ArrayXcd &);
	template double Interpolate<double>(double, const Eigen::ArrayXd &, const Eigen::ArrayXd &);

	double GetDifferential(const Eigen::ArrayXd & intVar, int nr) {
		double dIntVar;
		if (intVar.size() <= 1) {
			dIntVar = 1.0;
		}
		else if (nr == 0) {
			dIntVar = intVar(1) - intVar(0);
		}
		else if (nr + 1 == intVar.size()) {
			dIntVar = intVar(intVar.size() - 1) - intVar(intVar.size() - 2);
		}
		else {
			dIntVar = 0.5 * (intVar(nr + 1) - intVar(nr - 1));
		}
		return dIntVar;
	}


};