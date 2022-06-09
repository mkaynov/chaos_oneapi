#pragma once

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

namespace stat_tool {
    static double mean(double *in, int num, int step) {
        double accum = 0;
        for (int i = 0; i < num; ++i) {
            accum += in[i * step];
        }
        return accum / num;
    }
    static double min(double *in, int num) {
        double accum = std::numeric_limits<double>::quiet_NaN();
        for (int i = 0; i < num; ++i) {
            if (not isnan(in[i])) {
                if (isnan(accum)) {
                    accum = in[i];
                    continue;
                }
                accum = accum < in[i] ? accum : in[i];
            }
        }
        return accum;
    }
    static double max(double *in, int num) {
        double accum = *in;
        for (int i = 1; i < num; ++i) {
            accum = accum > in[i] ? accum : in[i];
        }
        return accum;
    }
}

