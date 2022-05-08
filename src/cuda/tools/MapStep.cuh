

#include "types.h"


__device__
void StepMAP(double* val, diffSysFunc diffFunc, double* params, double* arg, const int32_t dimension);

__device__
void StepMAPVAR(double* val, diffSysFuncVar diffFuncVar, double* params, double* arg, const int32_t dimension, double* mainTraj);