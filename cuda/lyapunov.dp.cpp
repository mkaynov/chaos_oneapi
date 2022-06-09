#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "lyapunov.dp.hpp"

void deviceLESpectrumDverk_(double* mainTrajectory, double* slaveTrajectories, const int32_t dimension, const ModelType type, diffSysFunc diffFunc, double* params, double eps, double* lyapExp, int32_t expsNum,
	double step, double timeSkip, double timeSkipClP, double calcTime, double addSkipSteps, double* arg, double* k1, double* k2, double* k3, double* k4, double* k5, double* k6, double* k7, double* k8, double* projSum) {

	int32_t skipCount = 0;

	if (type == ModelType::flow) {
		for (double t = 0; t < timeSkip; t += step) // Skip points to reach the attractor
			dverkStep(mainTrajectory, dimension, diffFunc, params, step, arg, k1, k2, k3, k4, k5, k6, k7, k8);

		for (int32_t i = 0; i < expsNum; i++) {
			memcpy(&slaveTrajectories[i * dimension], mainTrajectory, dimension * sizeof(double));
			slaveTrajectories[i * dimension + i] += eps;
		}

		for (double t = 0; t < timeSkipClP; t += step) {
			dverkStep(mainTrajectory, dimension, diffFunc, params, step, arg, k1, k2, k3, k4, k5, k6, k7, k8);

			for (int32_t i = 0; i < expsNum; i++) {
				dverkStep(&slaveTrajectories[i * dimension], dimension, diffFunc, params, step, arg, k1, k2, k3, k4, k5, k6, k7, k8);
				for (int32_t j = 0; j < dimension; j++)
					slaveTrajectories[i * dimension + j] -= mainTrajectory[j];
			}

			ortVecs(slaveTrajectories, dimension, expsNum, projSum);
			normalizeVecs(slaveTrajectories, dimension, expsNum, eps);

			for (int32_t i = 0; i < expsNum; i++) {
				for (int32_t j = 0; j < dimension; j++)
					slaveTrajectories[i * dimension + j] += mainTrajectory[j];
			}
		}

		for (double t = 0; t < calcTime; t += step) {
			skipCount++;
			dverkStep(mainTrajectory, dimension, diffFunc, params, step, arg, k1, k2, k3, k4, k5, k6, k7, k8);
			for (int32_t i = 0; i < expsNum; i++) {
				dverkStep(&slaveTrajectories[i * dimension], dimension, diffFunc, params, step, arg, k1, k2, k3, k4, k5, k6, k7, k8);
			}

			if (skipCount == addSkipSteps) {
				for (int32_t i = 0; i < expsNum; i++) {
					for (int32_t j = 0; j < dimension; j++)
						slaveTrajectories[i * dimension + j] -= mainTrajectory[j];
				}
				ortVecs(slaveTrajectories, dimension, expsNum, projSum);

				for (int32_t i = 0; i < expsNum; i++)
                                        lyapExp[i] += sycl::log(
                                            vecNorm(
                                                &slaveTrajectories[i *
                                                                   dimension],
                                                dimension) /
                                            eps);

                                normalizeVecs(slaveTrajectories, dimension, expsNum, eps);

				for (int32_t i = 0; i < expsNum; i++) {
					for (int32_t j = 0; j < dimension; j++)
						slaveTrajectories[i * dimension + j] += mainTrajectory[j];
				}
				skipCount = 0;
			}
		}
	}
	
	if (type == ModelType::map) {
		for (int32_t t = 0; t < timeSkip; t++) // Skip points to reach the attractor
			StepMAP(mainTrajectory, diffFunc, params, arg, dimension);

		for (int32_t i = 0; i < expsNum; i++) {
			memcpy(&slaveTrajectories[i * dimension], mainTrajectory, dimension * sizeof(double));
			slaveTrajectories[i * dimension + i] += eps;
		}

		for (int32_t t = 0; t < timeSkipClP; t++) {
			StepMAP(mainTrajectory, diffFunc, params, arg, dimension);

			for (int32_t i = 0; i < expsNum; i++) {
				StepMAP(&slaveTrajectories[i * dimension], diffFunc, params, arg, dimension);
				for (int32_t j = 0; j < dimension; j++)
					slaveTrajectories[i * dimension + j] -= mainTrajectory[j];
			}

			ortVecs(slaveTrajectories, dimension, expsNum, projSum);
			normalizeVecs(slaveTrajectories, dimension, expsNum, eps);

			for (int32_t i = 0; i < expsNum; i++) {
				for (int32_t j = 0; j < dimension; j++)
					slaveTrajectories[i * dimension + j] += mainTrajectory[j];
			}
		}

		for (int32_t t = 0; t < calcTime; t++) {
			skipCount++;
			StepMAP(mainTrajectory, diffFunc, params, arg, dimension);
			for (int32_t i = 0; i < expsNum; i++) {
				StepMAP(&slaveTrajectories[i * dimension], diffFunc, params, arg, dimension);
			}


			for (int32_t i = 0; i < expsNum; i++) {
				for (int32_t j = 0; j < dimension; j++)
					slaveTrajectories[i * dimension + j] -= mainTrajectory[j];
			}
			ortVecs(slaveTrajectories, dimension, expsNum, projSum);

			for (int32_t i = 0; i < expsNum; i++)
                                lyapExp[i] += sycl::log(
                                    vecNorm(&slaveTrajectories[i * dimension],
                                            dimension) /
                                    eps);

                        normalizeVecs(slaveTrajectories, dimension, expsNum, eps);

			for (int32_t i = 0; i < expsNum; i++) {
				for (int32_t j = 0; j < dimension; j++)
					slaveTrajectories[i * dimension + j] += mainTrajectory[j];
			}
			skipCount = 0;
		}
	}

	for (int32_t i = 0; i < expsNum; i++)
		lyapExp[i] = lyapExp[i] / calcTime;
}



void deviceLESpectrumVARDverk_(double* mainTrajectory, double* slaveTrajectories, const int32_t dimension, const ModelType type, diffSysFunc diffFunc, diffSysFuncVar diffFuncVar, double* params, double eps, double* lyapExp, int32_t expsNum,
	double step, double timeSkip, double timeSkipClP, double calcTime, double addSkipSteps, double* arg, double* k1, double* k2, double* k3, double* k4, double* k5, double* k6, double* k7, double* k8, double* projSum) {

	int32_t skipCount = 0;

	if (type == ModelType::flow) {
		for (double t = 0; t < timeSkip; t += step) // Skip points to reach the attractor
			dverkStep(mainTrajectory, dimension, diffFunc, params, step, arg, k1, k2, k3, k4, k5, k6, k7, k8);

		memset(slaveTrajectories, 0., expsNum * dimension * sizeof(double));

		for (int32_t i = 0; i < expsNum; i++)
			slaveTrajectories[i * dimension + i] += eps;

		for (double t = 0; t < timeSkipClP; t += step) {

			for (int32_t i = 0; i < expsNum; i++) {
				dverkStepVarMat(&slaveTrajectories[i * dimension], dimension, diffFuncVar, params, step, arg, mainTrajectory, k1, k2, k3, k4, k5, k6, k7, k8);
			}
			dverkStep(mainTrajectory, dimension, diffFunc, params, step, arg, k1, k2, k3, k4, k5, k6, k7, k8);

			ortVecs(slaveTrajectories, dimension, expsNum, projSum);
			normalizeVecs(slaveTrajectories, dimension, expsNum, eps);
		}

		for (double t = 0; t < calcTime; t += step) {
			skipCount++;
			for (int32_t i = 0; i < expsNum; i++)
				dverkStepVarMat(&slaveTrajectories[i * dimension], dimension, diffFuncVar, params, step, arg, mainTrajectory, k1, k2, k3, k4, k5, k6, k7, k8);

			dverkStep(mainTrajectory, dimension, diffFunc, params, step, arg, k1, k2, k3, k4, k5, k6, k7, k8);

			if (skipCount == addSkipSteps) {
				ortVecs(slaveTrajectories, dimension, expsNum, projSum);

				for (int32_t i = 0; i < expsNum; i++)
                                        lyapExp[i] += sycl::log(
                                            vecNorm(
                                                &slaveTrajectories[i *
                                                                   dimension],
                                                dimension) /
                                            eps);

                                normalizeVecs(slaveTrajectories, dimension, expsNum, eps);
				skipCount = 0;
			}
		}
	}

	if (type == ModelType::map) {
		for (double t = 0; t < timeSkip; t += 1) // Skip points to reach the attractor
			StepMAP(mainTrajectory, diffFunc, params, arg, dimension);

		memset(slaveTrajectories, 0., expsNum * dimension * sizeof(double));

		for (int32_t i = 0; i < expsNum; i++)
			slaveTrajectories[i * dimension + i] += eps;


		for (double t = 0; t < timeSkipClP; t += 1) {

			for (int32_t i = 0; i < expsNum; i++) 
				StepMAPVAR(&slaveTrajectories[i * dimension], diffFuncVar, params, arg, dimension, mainTrajectory);
			
			StepMAP(mainTrajectory, diffFunc, params, arg, dimension);

			ortVecs(slaveTrajectories, dimension, expsNum, projSum);
			normalizeVecs(slaveTrajectories, dimension, expsNum, eps);
		}

		for (double t = 0; t < calcTime; t += 1) {
			skipCount++;
			for (int32_t i = 0; i < expsNum; i++)
				StepMAPVAR(&slaveTrajectories[i * dimension], diffFuncVar, params, arg, dimension, mainTrajectory);

			StepMAP(mainTrajectory, diffFunc, params, arg, dimension);

			ortVecs(slaveTrajectories, dimension, expsNum, projSum);

			for (int32_t i = 0; i < expsNum; i++)
                                lyapExp[i] += sycl::log(
                                    vecNorm(&slaveTrajectories[i * dimension],
                                            dimension) /
                                    eps);

                        normalizeVecs(slaveTrajectories, dimension, expsNum, eps);
			skipCount = 0;
			
		}
	}

	for (int32_t i = 0; i < expsNum; i++) 
		lyapExp[i] = lyapExp[i] / calcTime;
}


void deviceLESpectrumQR_(double* mainTrajectory, double* slaveTrajectories, const int32_t dimension, diffSysFunc diffFunc, diffSysFuncVar linearQ,
	double* params, double eps, double* lyapExp, int32_t expsNum,
	double step, double timeSkip, double timeSkipClP, double calcTime, double addSkipSteps,
	double* U, double* R, double* Q,
	double* arg, double* k1, double* k2, double* k3, double* k4, double* k5, double* k6, double* k7, double* k8, double* projSum,
	double* argQ, double* k1Q, double* k2Q, double* k3Q, double* k4Q, double* k5Q, double* k6Q, double* k7Q, double* k8Q) {

	for (double t = 0; t < timeSkip; t += step) // Skip points to reach the attractor
		dverkStep(mainTrajectory, dimension, diffFunc, params, step, arg, k1, k2, k3, k4, k5, k6, k7, k8);

	memset(Q, 0, dimension * dimension * sizeof(double));
	for (int32_t i = 0; i < expsNum; i++)
		for (int32_t j = 0; j < dimension; j++)
			Q[i * dimension + i] = 1;

	for (double t = timeSkip; t < timeSkip + calcTime; t += step) {
		dverkStep(mainTrajectory, dimension, diffFunc, params, step, arg, k1, k2, k3, k4, k5, k6, k7, k8);
		dverkStepVarMat(Q, dimension * dimension, linearQ, params, step, argQ, mainTrajectory, k1Q, k2Q, k3Q, k4Q, k5Q, k6Q, k7Q, k8Q);

		memcpy(U, Q, dimension * dimension * sizeof(double));

		qrDecomposition(R, Q, U, dimension, expsNum, projSum);

		for (int32_t i = 0; i < expsNum; i++)
                        lyapExp[i] += sycl::log(sycl::fabs(R[i * expsNum + i]));
        }

	for (int32_t i = 0; i < expsNum; i++) {
		lyapExp[i] = lyapExp[i] / calcTime;
	}
}


void kernLEDverk_(bool is_qr, bool is_var, double* inits, double* closePoints, const int32_t dimension, const ModelType type, diffSysFunc diffFunc, diffSysFuncVar diffFuncVar, diffSysFuncVar modelJacobian,
	double* params, int32_t paramsDim, double eps, double step, double* lyapExps,
	int32_t expsNum, double timeSkip, double timeSkipClP, double calcTime, int32_t knodsNum, double addSkipSteps,
	double* U, double* R, double* Q,
	double* arg, double* k1, double* k2, double* k3, double* k4, double* k5, double* k6, double* k7, double* k8, double* projSum,
	double* argQ, double* k1Q, double* k2Q, double* k3Q, double* k4Q, double* k5Q, double* k6Q, double* k7Q, double* k8Q,
	sycl::nd_item<3> item_ct1, const sycl::stream &stream_ct1) {

        int32_t index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                        item_ct1.get_local_id(2);
        int32_t stride = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);

    if (!is_qr) {
        if(!is_var) {
            for (int32_t i = index; i < knodsNum; i += stride) {
                int32_t intInd = index * dimension;
                deviceLESpectrumDverk_(&inits[i * dimension], &closePoints[i * dimension * expsNum], dimension, type, diffFunc, &params[i * paramsDim], eps, &lyapExps[i * expsNum], expsNum, step, timeSkip, timeSkipClP, calcTime, addSkipSteps,
                    &arg[intInd], &k1[intInd], &k2[intInd], &k3[intInd], &k4[intInd], &k5[intInd], &k6[intInd], &k7[intInd], &k8[intInd], &projSum[intInd]);
            }
        }
        else {
            if (diffFuncVar == nullptr) {
                stream_ct1 << "Model doesn't have VAR function. Please add it "
                              "to the model\n";
                return;
            }
            for (int32_t i = index; i < knodsNum; i += stride) {
                int32_t intInd = index * dimension;
                deviceLESpectrumVARDverk_(&inits[i * dimension], &closePoints[i * dimension * expsNum], dimension, type, diffFunc, diffFuncVar, &params[i * paramsDim], eps, &lyapExps[i * expsNum], expsNum, step, timeSkip, timeSkipClP, calcTime, addSkipSteps,
                    &arg[intInd], &k1[intInd], &k2[intInd], &k3[intInd], &k4[intInd], &k5[intInd], &k6[intInd], &k7[intInd], &k8[intInd], &projSum[intInd]);

            }
        }
	}
	else {
		if (modelJacobian == nullptr) {
                        stream_ct1 << "Model doesn't have Jacobian function. "
                                      "Please add it to the model\n";
                        return;
		}
		if (type == ModelType::map) {
                        stream_ct1 << "Map model cant use Jacobian method. "
                                      "Please choose another method\n";
                        return;
		}
		for (int32_t i = index; i < knodsNum; i += stride) {
			int32_t intInd = index * dimension;
			deviceLESpectrumQR_(&inits[i * dimension], &closePoints[i * dimension * expsNum], dimension, diffFunc, modelJacobian,
				&params[i * paramsDim], eps, &lyapExps[i * expsNum], expsNum, step, timeSkip, timeSkipClP, calcTime, addSkipSteps,
				&U[intInd * dimension], &R[intInd * dimension], &Q[intInd * dimension],
				&arg[intInd], &k1[intInd], &k2[intInd], &k3[intInd], &k4[intInd], &k5[intInd], &k6[intInd], &k7[intInd], &k8[intInd], &projSum[intInd],
				&argQ[intInd * dimension], &k1Q[intInd * dimension], &k2Q[intInd * dimension], &k3Q[intInd * dimension],
				&k4Q[intInd* dimension], &k5Q[intInd* dimension], &k6Q[intInd * dimension], &k7Q[intInd * dimension], &k8Q[intInd * dimension]);

		}
	}
}


void cudaLyapunov(int bn, int bs, size_t taskNum, bool isQr, bool isVar, ModelType modelType, double *inits, double *closePoints,
                  const int32_t dimension, diffSysFunc diffFunc,
                  diffSysFuncVar diffFuncVar, diffSysFuncVar diffFuncVarQ,
                  double *params, int32_t paramsDim, double eps, double integrationStep, double *lyapExps,
                  int32_t expsNum, double timeSkip, double timeSkipClP, double calcTime,
                  double addSkipSteps, double *arg, double* U, double* R, double* Q, double *k1, double *k2, double *k3,
                  double *k4, double *k5, double *k6, double *k7, double *k8,
                  double *argQ, double *k1Q, double *k2Q, double *k3Q,
                  double *k4Q, double *k5Q, double *k6Q, double *k7Q, double *k8Q, double *projSum) {

    bn = 32;
    bs = 32;
                /*
                DPCT1049:0: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                                sycl::stream stream_ct1(64 * 1024, 80, cgh);

                                cgh.parallel_for(
                                    sycl::nd_range<3>(
                                        sycl::range<3>(1, 1, bn) *
                                            sycl::range<3>(1, 1, bs),
                                        sycl::range<3>(1, 1, bs)),
                                    [=](sycl::nd_item<3> item_ct1) {
                                                    kernLEDverk_(
                                                        isQr, isVar, inits,
                                                        closePoints, dimension,
                                                        modelType, diffFunc,
                                                        diffFuncVar,
                                                        diffFuncVarQ, params,
                                                        paramsDim, eps,
                                                        integrationStep,
                                                        lyapExps, expsNum,
                                                        timeSkip, timeSkipClP,
                                                        calcTime, taskNum,
                                                        addSkipSteps, U, R, Q,
                                                        arg, k1, k2, k3, k4, k5,
                                                        k6, k7, k8, projSum,
                                                        argQ, k1Q, k2Q, k3Q,
                                                        k4Q, k5Q, k6Q, k7Q, k8Q,
                                                        item_ct1, stream_ct1);
                                    });
                });
}

