#include "TaskLyapunov.h"
#include "singleton.h"
#include "cuda/lyapunov.cuh"
#include "RaiiTimer.h"

std::vector<std::vector<double>> &TaskLyapunov::accessResult() {
    return mResults;
}

int TaskLyapunov::init(std::shared_ptr<Parameters> paramsBase, const std::string &modelName) {
    mParameters = std::dynamic_pointer_cast<LyapunovParameters>(paramsBase);
    ModelFactoryRegistry::Storage & storage = Singleton<ModelFactoryRegistry::Storage>::Instance();
    mModelPtr = &storage._data.at(modelName);
    mKNodsNum = mParameters->first_sweep_param[3] * mParameters->second_sweep_param[3];

    mKNodsGPUBatch = mKNodsNum / mParameters->num_GPUs;

    // TODO: calculate optimal block num somehow
    while (mBlockNum * mBlockSize > 1.5 * mKNodsGPUBatch) {
        if (mBlockNum >= 2) {
            mBlockNum /= 2;
            continue;
        }
        if (mBlockSize >= 2) {
            mBlockSize /= 2;
            continue;
        }
    }
    printf("block num %zu, block size %zu, total threads %llu, nodes num %d.\n", mBlockNum, mBlockSize, mBlockNum*mBlockSize, mKNodsNum);

    return 0;
}

int TaskLyapunov::allocate() {

    auto t = RaiiTimer(&mClocks.allocate);

    if (mModelPtr == nullptr) {
        return -1;
    }

    param1_delta = fabs(mParameters->first_sweep_param[2] - mParameters->first_sweep_param[1]) / (mParameters->first_sweep_param[3] - 1);
    param2_delta = fabs(mParameters->second_sweep_param[2] - mParameters->second_sweep_param[1]) / (mParameters->second_sweep_param[3] - 1);
    params = (double*)malloc(mKNodsNum * mModelPtr->_param_number * sizeof(double));
    int32_t paramCount = 0;
    for (int p1 = 0; p1 < mParameters->first_sweep_param[3]; p1++) {
        for (int p2 = 0; p2 < mParameters->second_sweep_param[3]; p2++) {
            double firstParam = mParameters->first_sweep_param[1] + p1 * param1_delta;
            double secondParam = mParameters->second_sweep_param[1] + p2 * param2_delta;
            for (int32_t i = 0; i < mModelPtr->_param_number; i++) {
                if (i == (int32_t)mParameters->first_sweep_param[0])
                    params[paramCount * mModelPtr->_param_number + i] = firstParam;
                else if (i == (int32_t)mParameters->second_sweep_param[0])
                    params[paramCount * mModelPtr->_param_number + i] = secondParam;
                else
                    params[paramCount * mModelPtr->_param_number + i] = mParameters->initial_model_params[i];
            }
            paramCount++;
        }
    }

    addSkipSteps = mParameters->skip_time_normalization / mParameters->integration_step;

    inits = (double*)malloc(mKNodsNum * mModelPtr->_dimension * sizeof(double));

    for (int32_t i = 0; i < mKNodsNum; i++) {
        memcpy(&inits[i * mModelPtr->_dimension], mParameters->initial_state.data(), mModelPtr->_dimension * sizeof(double));
    }

    lyapExps = (double*)malloc(mKNodsNum * mParameters->lyapunov_exponent_num * sizeof(double));
    paramsCuda = (double**)malloc(mParameters->num_GPUs * sizeof(double*));
    initsCuda = (double**)malloc(mParameters->num_GPUs * sizeof(double*));
    closePointsCuda = (double**)malloc(mParameters->num_GPUs * sizeof(double*));
    lyapExpsCuda = (double**)malloc(mParameters->num_GPUs * sizeof(double*));

    arg = (double**)malloc(mParameters->num_GPUs * sizeof(double*));
    k1 = (double**)malloc(mParameters->num_GPUs * sizeof(double*));
    k2 = (double**)malloc(mParameters->num_GPUs * sizeof(double*));
    k3 = (double**)malloc(mParameters->num_GPUs * sizeof(double*));
    k4 = (double**)malloc(mParameters->num_GPUs * sizeof(double*));
    k5 = (double**)malloc(mParameters->num_GPUs * sizeof(double*));
    k6 = (double**)malloc(mParameters->num_GPUs * sizeof(double*));
    k7 = (double**)malloc(mParameters->num_GPUs * sizeof(double*));
    k8 = (double**)malloc(mParameters->num_GPUs * sizeof(double*));

    U = (double**)malloc(mParameters->num_GPUs * sizeof(double*));
    R = (double**)malloc(mParameters->num_GPUs * sizeof(double*));
    Q = (double**)malloc(mParameters->num_GPUs * sizeof(double*));
    argQ = (double**)malloc(mParameters->num_GPUs * sizeof(double*));
    k1Q = (double**)malloc(mParameters->num_GPUs * sizeof(double*));
    k2Q = (double**)malloc(mParameters->num_GPUs * sizeof(double*));
    k3Q = (double**)malloc(mParameters->num_GPUs * sizeof(double*));
    k4Q = (double**)malloc(mParameters->num_GPUs * sizeof(double*));
    k5Q = (double**)malloc(mParameters->num_GPUs * sizeof(double*));
    k6Q = (double**)malloc(mParameters->num_GPUs * sizeof(double*));
    k7Q = (double**)malloc(mParameters->num_GPUs * sizeof(double*));
    k8Q = (double**)malloc(mParameters->num_GPUs * sizeof(double*));

    projSum = (double**)malloc(mParameters->num_GPUs * sizeof(double*));

    size_t cuda_size = mBlockNum * mBlockSize * mModelPtr->_dimension * sizeof(double);
    for (int32_t i = 0; i < mParameters->num_GPUs; i++) {
        cudaSetDevice(i);

        cudaMalloc(&paramsCuda[i], mKNodsGPUBatch * mModelPtr->_param_number * sizeof(double));
        cudaMemcpy(paramsCuda[i], &params[mKNodsGPUBatch * i * mModelPtr->_param_number], mKNodsGPUBatch * mModelPtr->_param_number * sizeof(double), cudaMemcpyHostToDevice);

        cudaMalloc(&initsCuda[i], mKNodsGPUBatch * mModelPtr->_dimension * sizeof(double));
        cudaMemcpy(initsCuda[i], &inits[mKNodsGPUBatch * i * mModelPtr->_dimension], mKNodsGPUBatch * mModelPtr->_dimension * sizeof(double), cudaMemcpyHostToDevice);

        cudaMalloc(&closePointsCuda[i], mParameters->lyapunov_exponent_num * mKNodsGPUBatch * mModelPtr->_dimension * sizeof(double));

        cudaMalloc(&lyapExpsCuda[i], mKNodsGPUBatch * mParameters->lyapunov_exponent_num * sizeof(double));

        cudaMalloc(&arg[i], cuda_size);
        cudaMalloc(&k1[i], cuda_size);
        cudaMalloc(&k2[i], cuda_size);
        cudaMalloc(&k3[i], cuda_size);
        cudaMalloc(&k4[i], cuda_size);
        cudaMalloc(&k5[i], cuda_size);
        cudaMalloc(&k6[i], cuda_size);
        cudaMalloc(&k7[i], cuda_size);
        cudaMalloc(&k8[i], cuda_size);
        cudaMalloc(&projSum[i], cuda_size);

        if (mParameters->is_qr) {
            cudaMalloc(&U[i], cuda_size * mModelPtr->_dimension);
            cudaMalloc(&R[i], cuda_size * mModelPtr->_dimension);
            cudaMalloc(&Q[i], cuda_size * mModelPtr->_dimension);

            cudaMalloc(&argQ[i], cuda_size * mModelPtr->_dimension);
            cudaMalloc(&k1Q[i], cuda_size * mModelPtr->_dimension);
            cudaMalloc(&k2Q[i], cuda_size * mModelPtr->_dimension);
            cudaMalloc(&k3Q[i], cuda_size * mModelPtr->_dimension);
            cudaMalloc(&k4Q[i], cuda_size * mModelPtr->_dimension);
            cudaMalloc(&k5Q[i], cuda_size * mModelPtr->_dimension);
            cudaMalloc(&k6Q[i], cuda_size * mModelPtr->_dimension);
            cudaMalloc(&k7Q[i], cuda_size * mModelPtr->_dimension);
            cudaMalloc(&k8Q[i], cuda_size * mModelPtr->_dimension);
        }
    }

    mResults.reserve(mKNodsNum);
    return 0;
}

int TaskLyapunov::execute() {

    auto t = RaiiTimer(&mClocks.execute);

    mResults.clear();

    for (int32_t i = 0; i < mParameters->num_GPUs; i++) {
        cudaSetDevice(i);
        diffSysFunc diff_func = nullptr;
        diffSysFuncVar diff_func_var = nullptr;
        diffSysFuncVar diff_func_Q = nullptr;

        mModelPtr->_model_func(&diff_func);
        if (mModelPtr->_func_var)
            mModelPtr->_func_var(&diff_func_var);
        if (mModelPtr->_func_Q)
            mModelPtr->_func_Q(&diff_func_Q);
        cudaLyapunov(mBlockNum, mBlockSize, mKNodsGPUBatch, mParameters->is_qr, mParameters->is_var, mModelPtr->_type, initsCuda[i], closePointsCuda[i], mModelPtr->_dimension,
                     diff_func, diff_func_var, diff_func_Q, paramsCuda[i], mModelPtr->_param_number, mParameters->epsilon, mParameters->integration_step,
                     lyapExpsCuda[i], mParameters->lyapunov_exponent_num, mParameters->skip_time_attractor, mParameters->skip_time_slave_trajectory, mParameters->total_time,
                     addSkipSteps, arg[i], U[i], R[i], Q[i], k1[i], k2[i], k3[i], k4[i], k5[i], k6[i], k7[i], k8[i], argQ[i],
                     k1Q[i], k2Q[i], k3Q[i], k4Q[i], k5Q[i], k6Q[i], k7Q[i], k8Q[i], projSum[i]);
    }

    for (int32_t i = 0; i < mParameters->num_GPUs; i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
        cudaMemcpy(&lyapExps[i * mKNodsGPUBatch * mParameters->lyapunov_exponent_num], lyapExpsCuda[i], mKNodsGPUBatch * mParameters->lyapunov_exponent_num * sizeof(double), cudaMemcpyDeviceToHost);
    }

    // TODO: can be used as a check
    //std::cout << "\nLyapunov exponents fw " << lyapExps[0] << " " << lyapExps[1] << " " << lyapExps[2] << std::endl;

    int32_t paramCount = 0;
    for (int p1=0; p1 < mParameters->first_sweep_param[3]; p1++) {
        for (int p2=0; p2 < mParameters->second_sweep_param[3]; p2++) {
            double firstParam = mParameters->first_sweep_param[1] + p1 * param1_delta;
            double secondParam = mParameters->second_sweep_param[1] + p2 * param2_delta;
            std::vector<double> res;
            res.reserve(2 + mParameters->lyapunov_exponent_num);
            res.push_back(firstParam);
            res.push_back(secondParam);

            for (int32_t i = 0; i < mParameters->lyapunov_exponent_num; i++) {
                res.push_back(lyapExps[paramCount * mParameters->lyapunov_exponent_num + i]);
            }
            mResults.push_back(std::move(res));
            paramCount++;
        }
    }

    mReady = true;
    return mKNodsNum;
}

int TaskLyapunov::clear() {

    auto t = RaiiTimer(&mClocks.clear);

    for (int32_t i = 0; i < mParameters->num_GPUs; i++) {
        cudaSetDevice(i);
        cudaFree(arg[i]);
        cudaFree(k1[i]);
        cudaFree(k2[i]);
        cudaFree(k3[i]);
        cudaFree(k4[i]);
        cudaFree(k5[i]);
        cudaFree(k6[i]);
        cudaFree(k7[i]);
        cudaFree(k8[i]);
        cudaFree(projSum[i]);
        cudaFree(paramsCuda[i]);
        cudaFree(initsCuda[i]);
        cudaFree(lyapExpsCuda[i]);
        cudaFree(closePointsCuda[i]);

        if (mParameters->is_qr) {
            cudaFree(U[i]);
            cudaFree(R[i]);
            cudaFree(Q[i]);

            cudaFree(argQ[i]);
            cudaFree(k1Q[i]);
            cudaFree(k2Q[i]);
            cudaFree(k3Q[i]);
            cudaFree(k4Q[i]);
            cudaFree(k5Q[i]);
            cudaFree(k6Q[i]);
            cudaFree(k7Q[i]);
            cudaFree(k8Q[i]);
        }
    }

    free(arg);
    free(k1);
    free(k2);
    free(k3);
    free(k4);
    free(k5);
    free(k6);
    free(k7);
    free(k8);
    free(projSum);
    free(paramsCuda);
    free(initsCuda);
    free(lyapExpsCuda);
    free(closePointsCuda);

    free(U);
    free(R);
    free(Q);

    free(argQ);
    free(k1Q);
    free(k2Q);
    free(k3Q);
    free(k4Q);
    free(k5Q);
    free(k6Q);
    free(k7Q);
    free(k8Q);


    free(inits);
    free(lyapExps);
    free(params);
    return 0;
}


