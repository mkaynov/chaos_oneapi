
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "MapStep.dp.hpp"

SYCL_EXTERNAL
void StepMAP(double* val, diffSysFunc diffFunc, double* params, double* arg, const int32_t dimension) {

	diffFunc(val, arg, params);
	for (int32_t ii = 0; ii < dimension; ii++)
		val[ii] = arg[ii];
}

SYCL_EXTERNAL
void StepMAPVAR(double* val, diffSysFuncVar diffFuncVar, double* params, double* arg, const int32_t dimension, double* mainTraj) {

	diffFuncVar(val, arg, params, mainTraj);
	for (int32_t ii = 0; ii < dimension; ii++)
		val[ii] = arg[ii];
}