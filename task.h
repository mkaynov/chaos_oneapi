#ifndef CHAOS_TASK_H
#define CHAOS_TASK_H

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "memory"
#include "tool_params.h"
#include "consts.h"
#include "model_factory.h"

class Task {
    struct Clocks {
        uint32_t allocate;
        uint32_t execute;
        uint32_t clear;
    };
public:
    Task() : mBlockSize(DEFAULT_BLOCK_SIZE), mBlockNum(DEFAULT_BLOCK_NUM), mClocks({ 0, 0, 0 }), mReady(false) {};

    virtual int init(std::shared_ptr<Parameters> paramsBase, const std::string &modelName) = 0;

    virtual int allocate() = 0;

    virtual int execute() = 0;

    virtual int clear() = 0;

    virtual bool isReady() final { return mReady; }

    virtual uint32_t getAllocationTime() { return mClocks.allocate; }

    virtual uint32_t getExecutionTime() { return mClocks.execute; }

    virtual std::vector<std::vector<double>> &accessResult() = 0;

    virtual void printTimeReport() final {
        printf("\n========TIME REPORT==========\n");
        printf(" allocation: %u\n", mClocks.allocate);
        printf(" execution:  %u\n", mClocks.execute);
        printf(" clear:      %u\n", mClocks.clear);
        printf(" Total:      %u\n", mClocks.allocate + mClocks.execute + mClocks.clear);
        printf("==============================\n\n");
    }

protected:
    static bool allocateDoublesCuda(double **ptr, size_t size) try {
        /*
        DPCT1003:0: Migrated API does not return error code. (*, 0) is inserted.
        You may need to rewrite this code.
        */
        if ((*ptr =
            sycl::malloc_device<double>(size, dpct::get_default_queue()),
            0)) {
            printf("failed to allocate %llu Mbytes on CUDA\n", size * sizeof(double) / 1024 / 1024);
            return false;
        }
        return true;
    }
    catch (sycl::exception const &exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
        std::exit(1);
    };
    size_t mBlockSize;
    size_t mBlockNum;
    Clocks mClocks;
    std::vector<std::string> mResultsStr;
    std::vector <std::vector<double>> mResults;
    bool mReady;

    friend int32_t saveCalculatedData(ToolID tool_id, const std::string &modelName, const Parameters *params,
        const Task*);
};

#endif //CHAOS_TASK_H
