#pragma once

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "D:\Work\Startup\oneAPI_DPC++\chaos_migration\types.h"

SYCL_EXTERNAL
void dverkStep(double* val, const int32_t dimension, diffSysFunc diffFunc, double* params, double step,
	double* arg, double* k1, double* k2, double* k3, double* k4, double* k5, double* k6, double* k7, double* k8);

SYCL_EXTERNAL
void dverkStepVarMat(double* val, const int32_t dimension, diffSysFuncVar diffFunc, double* params, double step,
	double* arg, double* mainTraj, double* k1, double* k2, double* k3, double* k4, double* k5, double* k6, double* k7, double* k8);
