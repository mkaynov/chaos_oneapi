

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "D:\Work\Startup\oneAPI_DPC++\chaos_migration\types.h"

SYCL_EXTERNAL
void StepMAP(double* val, diffSysFunc diffFunc, double* params, double* arg, const int32_t dimension);

SYCL_EXTERNAL 
void StepMAPVAR(double* val, diffSysFuncVar diffFuncVar, double* params, double* arg, const int32_t dimension, double* mainTraj);