#pragma once
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "model_factory.h"


void dopriStep(double* val, int dimension, diffSysFunc diffFunc, double* params, double& step, double maxStep, double tolLocErr, double tolGlobErr,
	double* spareVal, double* ySti, double* arg, double* k1, double* k2, double* k3, double* k4, double* k5, double* k6, double& time, double& facOld);