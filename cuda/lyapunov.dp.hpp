#pragma once

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "model_factory.h"
#include "tools/dverk.dp.hpp"
#include "tools/MapStep.dp.hpp"
#include "tools/linalg.dp.hpp"

void cudaLyapunov(int bn, int bs, size_t taskNum, bool isQr, bool isVar, ModelType modelType, double *inits,
                  double *closePoints, const int32_t dimension, diffSysFunc diffFunc, diffSysFuncVar diffFuncVar, diffSysFuncVar diffFuncVarQ,
                  double *params, int32_t paramsDim, double eps, double integrationStep, double *lyapExps,
                  int32_t expsNum, double timeSkip, double timeSkipClP, double calcTime,
                  double addSkipSteps, double *arg, double* U, double* R, double* Q, double *k1, double *k2, double *k3,
                  double *k4, double *k5, double *k6, double *k7, double *k8,
                  double *argQ, double *k1Q, double *k2Q, double *k3Q,
                  double *k4Q, double *k5Q, double *k6Q, double *k7Q, double *k8Q, double *projSum);