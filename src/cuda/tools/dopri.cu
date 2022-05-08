#pragma once 

#include "dopri.cuh"

__device__
void dopriStep(double* val, int dimension, diffSysFunc diffFunc, double* params, double& step, double maxStep, double tolLocErr, double tolGlobErr,
	double* spareVal, double* ySti, double* arg, double* k1, double* k2, double* k3, double* k4, double* k5, double* k6, double& time, double& facOld) {
	double safeFac = 0.9;
	double fac1 = 0.2;
	double fac2 = 10.;
	double beta = 0.04;

	//coefficients of the Runge-Kutta scheme
	const double c2 = 0.2;
	const double c3 = 0.3;
	const double c4 = 0.8;
	const double c5 = 8. / 9.;
	const double a21 = 0.2;
	const double a31 = 3. / 40.;
	const double a32 = 9. / 40.;
	const double a41 = 44. / 45.;
	const double a42 = -56. / 15.;
	const double a43 = 32. / 9.;
	const double a51 = 19372. / 6561.;
	const double a52 = -25360. / 2187.;
	const double a53 = 64448. / 6561.;
	const double a54 = -212. / 729.;
	const double a61 = 9017. / 3168.;
	const double a62 = -355. / 33.;
	const double a63 = 46732. / 5247.;
	const double a64 = 49. / 176.;
	const double a65 = -5103. / 18656.;
	const double a71 = 35. / 384.;
	const double a73 = 500. / 1113.;
	const double a74 = 125. / 192.;
	const double a75 = -2187. / 6784.;
	const double a76 = 11. / 84.;
	const double e1 = 71. / 57600.;
	const double e3 = -71. / 16695.;
	const double e4 = 71. / 1920.;
	const double e5 = -17253. / 339200.;
	const double e6 = 22. / 525.;
	const double e7 = -1. / 40.;

	double expo1 = 0.2 - beta * 0.75;
	double facc1 = 1. / fac1;
	double facc2 = 1. / fac2;

	double fac11 = 0;
	double fac = 0;
	double err = 0;
	double sk = 0;
	double xph = 0;

	bool isReject = false;

stepRejected:

	maxStep = fabs(maxStep);
	double denom = 0;
	double newStep = 0;
	int32_t memsize = sizeof(double) * dimension;
	memcpy(spareVal, val, memsize);
	if (time == 0) {
		facOld = 1e-4;
		diffFunc(val, k1, params);
	}

	for (int i = 0; i < dimension; i++) {
		arg[i] = val[i] + step * a21 * k1[i];
	}
	diffFunc(arg, k2, params);

	for (int i = 0; i < dimension; i++) {
		arg[i] = val[i] + step * (a31 * k1[i] + a32 * k2[i]);
	}
	diffFunc(arg, k3, params);
	for (int i = 0; i < dimension; i++) {
		arg[i] = val[i] + step * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
	}
	diffFunc(arg, k4, params);
	for (int i = 0; i < dimension; i++) {
		arg[i] = val[i] + step * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
	}
	diffFunc(arg, k5, params);
	for (int i = 0; i < dimension; i++) {
		ySti[i] = val[i] + step * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
	}
	diffFunc(ySti, k6, params);
	for (int i = 0; i < dimension; i++) {
		arg[i] = val[i] + step * (a71 * k1[i] + a73 * k3[i] + a74 * k4[i] + a75 * k5[i] + a76 * k6[i]);
	}
	diffFunc(arg, k2, params);
	for (int i = 0; i < dimension; i++) {
		k4[i] = (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k2[i]) * step;
	}
	err = 0.;
	sk = 0.;
	for (int i = 0; i < dimension; i++) {
		sk = tolLocErr + tolGlobErr * (fabs(val[i]) > fabs(arg[i]) ? fabs(val[i]) : fabs(arg[i]));  //std::max(abs(val[i]), abs(arg[i]))
		err = err + pow((k4[i] / sk), 2);
	}
	err = sqrt(err / dimension);
	fac11 = pow(err, expo1);
	fac = fac11 / pow(facOld, beta); //std::max(facc2, std::min(facc1, fac / safeFac))
	fac = facc2 > (facc1 < (fac / safeFac) ? facc1 : (fac / safeFac)) ? facc2 : (facc1 < (fac / safeFac) ? facc1 : (fac / safeFac));// fac1 <= newStep / step <= fac2 (//fac = std::max(facc2, std::min(facc1, fac / safeFac)); )
	newStep = step / fac;
	if (err <= 1.) {					//step accepted
		facOld = err > 1e-4 ? err : 1e-4; //std::max(err, 1.0E-4)
		for (int i = 0; i < dimension; i++) {
			k1[i] = k2[i];
			val[i] = arg[i];
		}
		if (fabs(newStep) > maxStep)
			newStep = maxStep;
		if (isReject)
			newStep = fabs(newStep) < fabs(step) ? fabs(newStep) : fabs(step); // std::min(abs(newStep), abs(step))
		isReject = false;
		time += step;
		step = newStep;
	}
	else {													//step is rejected=
		newStep = step / (facc1 < (fac11 / safeFac) ? facc1 : (fac11 / safeFac)); // std::min(facc1, fac11 / safeFac)
		isReject = true;
		memcpy(val, spareVal, memsize);
		step = newStep;
		goto stepRejected;
	}
}