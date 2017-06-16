#include "SecondOrderNLTMM.h"
#include <ctime>

void TestNonlinearTmm() {
	double wl = 532e-9;
	TMM::Polarization pol = TMM::P_POL;
	TMM::NonlinearTmmMode mode = TMM::MODE_INCIDENT;
	TMM::Material prismN(1.5);
	TMM::Material crystalN(TMM::dcomplex(0.05400722021660654, 3.428989169675091));
	TMM::Material airN(1.0);
	double metalD = 50e-9;

	TMM::NonlinearTMM tmm;
	tmm.AddLayer(TMM::INF, &prismN);
	tmm.AddLayer(metalD, &crystalN);
	tmm.AddLayer(TMM::INF, &airN);

	tmm.SetParam(TMM::PARAM_WL, wl);
	tmm.SetParam(TMM::PARAM_POL, (int)pol);
	tmm.SetParam(TMM::PARAM_MODE, (int)mode);
	tmm.SetParam(TMM::PARAM_I0, 1.0);
	//tmm.SetParam(TMM::PARAM_OVERRIDE_E0, true);
	//tmm.SetParam(TMM::PARAM_E0, 1.0);

	//for (double beta = 1.01; beta < 1.5; beta += 0.0000001) {
	/*
	for (int i = 0; i < 2; i++) {
		double beta = 0.3;
		std::cout << "beta " << beta << std::endl << std::endl;
		tmm.SetParam(TMM::PARAM_BETA, beta);
		tmm.Solve();
		tmm.GetPowerFlows();
	}
	*/

	Eigen::ArrayXd betas = Eigen::ArrayXd::LinSpaced(1000, 0.0, 1.49);
	Eigen::Map<Eigen::ArrayXd> betasMap(&betas(0), betas.size());

	for (int i = 0; i < 5000; i++) {
		tmm.Sweep(TMM::PARAM_BETA, betasMap);
	}

	
	/*
	tmm.SetParam(TMM::PARAM_BETA, 0.0);
	tmm.Solve();
	
	
	ArrayXd zs = ArrayXd::LinSpaced(1000, -100e-9, 100e-9);
	Eigen::Map<ArrayXd> zsMap(&zs(0), zs.size());
	
	for (int i = 0; i < 50000; i++) {
		tmm.Solve();
		tmm.GetFields(zsMap);
	}
	/*

	/*
	ArrayXd xs = ArrayXd::LinSpaced(500, -100e-9, 100e-9);
	Eigen::Map<ArrayXd> xsMap(&xs(0), xs.size());

	ArrayXd zs = ArrayXd::LinSpaced(510, -200e-9, 200e-9);
	Eigen::Map<ArrayXd> zsMap(&zs(0), zs.size());

	for (int i = 0; i < 10000; i++) {
		tmm.Solve();
		TMM::FieldsZX *ptr = tmm.GetFields2D(xsMap, zsMap);
		delete ptr;
	}
	*/
	
}

void TestSecondOrderNLTMM() {
	double wlP1 = 400e-9;
	double wlP2 = 800e-9;
	TMM::Polarization polP1 = TMM::S_POL;
	TMM::Polarization polP2 = TMM::S_POL;
	TMM::Polarization polGen = TMM::S_POL;
	double betaP1 = 0.2;
	double betaP2 = 0.2;
	TMM::dcomplex overrideE0P1 = 1.0;
	TMM::dcomplex overrideE0P2 = 1.0;

	TMM::SecondOrderNLTMM tmm;
	tmm.SetProcess(TMM::SPDC);
	tmm.SetDeltaThetaSpdc(0.00872665);
	tmm.SetDeltaWlSpdc(2.5e-9);
	tmm.SetSolidAngleSpdc(7.61543549467e-05);

	TMM::NonlinearTMM *tmmP1 = tmm.GetP1();
	TMM::NonlinearTMM *tmmP2 = tmm.GetP2();
	TMM::NonlinearTMM *tmmGen = tmm.GetGen();

	tmmP1->SetParam(TMM::PARAM_WL, wlP1);
	tmmP1->SetParam(TMM::PARAM_BETA, betaP1);
	tmmP1->SetPolarization(polP1);
	tmmP1->SetOverrideE0(true);
	tmmP1->SetParam(TMM::PARAM_E0, overrideE0P1);

	tmmP2->SetParam(TMM::PARAM_WL, wlP2);
	tmmP2->SetParam(TMM::PARAM_BETA, betaP2);
	tmmP2->SetPolarization(polP2);
	tmmP2->SetOverrideE0(true);
	tmmP2->SetParam(TMM::PARAM_E0, overrideE0P2);

	tmmGen->SetPolarization(polGen);


	//  Waves
	tmmP1->GetWave()->SetWaveType(TMM::GAUSSIANWAVE);
	tmmP2->GetWave()->SetWaveType(TMM::SPDCWAVE);

	tmmP1->GetWave()->SetNPointsInteg(500);
	tmmP2->GetWave()->SetNPointsInteg(500);


	Eigen::Array4d wls;
	Eigen::Array4d ns;

	wls << 400e-9, 800e-9, 801e-9, 1500e-9;
	ns << 1.54, 1.54, 1.53, 1.53;

	TMM::Material prismN(1.5);
	TMM::Material crystalN(wls, ns);
	TMM::Material airN(1.0);
	double crystalD = 50e-9;

	TMM::Chi2Tensor &chi2 = crystalN.chi2;
	chi2.SetD(2, 2, 1e-12);

	tmm.AddLayer(TMM::INF, &prismN);
	tmm.AddLayer(crystalD, &crystalN);
	tmm.AddLayer(TMM::INF, &airN);

	/*
	Eigen::VectorXd betas = Eigen::VectorXd::LinSpaced(1000000, 0.2, 0.7);
	for (int i = 0; i < betas.size(); i++) {
	tmm.SetBetaP1(betas[i]);
	tmm.SetBetaP2(betas[i]);
	tmm.Solve();
	}
	*/

	//tmm.Solve();

	/*
	ArrayXd betas = ArrayXd::LinSpaced(1000, 0.0, 0.99);
	Eigen::Map<ArrayXd> betasMap(&betas(0), betas.size());
	for (int i = 0; i < 1000; i++) {
		tmm.Sweep(TMM::PARAM_BETA, betasMap, betasMap);
	}
	*/

	
	Eigen::ArrayXd xs = Eigen::ArrayXd::LinSpaced(200, -100e-9, 100e-9);
	Eigen::Map<Eigen::ArrayXd> xsMap(&xs(0), xs.size());

	Eigen::ArrayXd zs = Eigen::ArrayXd::LinSpaced(200, -200e-9, 200e-9);
	Eigen::Map<Eigen::ArrayXd> zsMap(&zs(0), zs.size());

	/*
	for (int i = 0; i < 10000; i++) {
	tmm.Solve();
	TMM::FieldsZX *ptr = tmm.GetGen()->GetFields2D(xsMap, zsMap);
	delete ptr;
	}
	*/

	TMM::FieldsZX *ptr = tmm.WaveGetFields2D(xsMap, zsMap);
	delete ptr;
}

int main() {
	//TestNonlinearTmm();
	TestSecondOrderNLTMM();
	
	std::cout << double(clock()) / CLOCKS_PER_SEC << std::endl;
	std::cout << "Done" << std::endl;
	//std::getchar();
	return 0;
}