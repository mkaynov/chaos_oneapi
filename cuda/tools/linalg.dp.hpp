
#pragma once

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "D:\Work\Startup\oneAPI_DPC++\chaos_migration\types.h"

double dotProduct(double* firstVec, double* secondVec, int32_t length);

SYCL_EXTERNAL
double vecNorm(double* vec, int32_t length);


double getAngle(double* vec1, double* vec2, double dimension);

SYCL_EXTERNAL
void ortVecs(double* vecs, int32_t dimension, int32_t numVecs, double* projSum);

SYCL_EXTERNAL
void normalizeVecs(double* vecs, int32_t dimension, int32_t numVec, double eps);


void matrixTranspose(double* matr, int32_t rows, int32_t cols);


void matrixMult(double* res, double* A, int32_t rowsA, int32_t colsA, double* B, int32_t rowsB, int32_t colsB);

SYCL_EXTERNAL
void qrDecomposition(double* R, double* Q, double* A, int32_t rows, int32_t cols, double* projSum);


double dotProductColMaj(double* firstVec, double* secondVec, int32_t rows, int32_t cols);


double vecNormColMaj(double* vec, int32_t rows, int32_t cols);