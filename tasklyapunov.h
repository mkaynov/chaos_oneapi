
#ifndef CHAOS_TASKLYAPUNOV_H
#define CHAOS_TASKLYAPUNOV_H

#include "Task.h"


class TaskLyapunov : public Task {
public:

    std::vector <std::vector<double>> &accessResult() override;

    int init(std::shared_ptr <Parameters> paramsBase, const std::string &modelName) override;

    int allocate() override;

    int execute() override;;

    int clear() override;

private:
    size_t mKNodsNum, mKNodsGPUBatch;
    double addSkipSteps;
    double param1_delta;
    double param2_delta;
    std::shared_ptr <LyapunovParameters> mParameters;
    ModelFactoryRegistry::ModelData *mModelPtr;

    double *lyapExps;
    double **paramsCuda;
    double **initsCuda;
    double **closePointsCuda;
    double **lyapExpsCuda;

    double **arg;
    double **k1;
    double **k2;
    double **k3;
    double **k4;
    double **k5;
    double **k6;
    double **k7;
    double **k8;
    double **argQ;
    double **k1Q;
    double **k2Q;
    double **k3Q;
    double **k4Q;
    double **k5Q;
    double **k6Q;
    double **k7Q;
    double **k8Q;
    double **U;
    double **R;
    double **Q;
    double **projSum;
    double *inits;
    double *params;
};

#endif //CHAOS_TASKLYAPUNOV_H
