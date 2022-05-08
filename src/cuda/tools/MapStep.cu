
#include "MapStep.cuh"


__device__
void StepMAP(double* val, diffSysFunc diffFunc, double* params, double* arg, const int32_t dimension) {

	diffFunc(val, arg, params);
	for (int32_t ii = 0; ii < dimension; ii++)
		val[ii] = arg[ii];
}

__device__
void StepMAPVAR(double* val, diffSysFuncVar diffFuncVar, double* params, double* arg, const int32_t dimension, double* mainTraj) {

	diffFuncVar(val, arg, params, mainTraj);
	for (int32_t ii = 0; ii < dimension; ii++)
		val[ii] = arg[ii];
}