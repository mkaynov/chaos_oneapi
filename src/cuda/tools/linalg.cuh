
#pragma once 

#include "types.h"

__device__
double dotProduct(double* firstVec, double* secondVec, int32_t length);

__device__
double vecNorm(double* vec, int32_t length);

__device__
double getAngle(double* vec1, double* vec2, double dimension);

__device__
void ortVecs(double* vecs, int32_t dimension, int32_t numVecs, double* projSum);

__device__
void normalizeVecs(double* vecs, int32_t dimension, int32_t numVec, double eps);

__device__
void matrixTranspose(double* matr, int32_t rows, int32_t cols);

__device__
void matrixMult(double* res, double* A, int32_t rowsA, int32_t colsA, double* B, int32_t rowsB, int32_t colsB);

__device__
void qrDecomposition(double* R, double* Q, double* A, int32_t rows, int32_t cols, double* projSum);

__device__
double dotProductColMaj(double* firstVec, double* secondVec, int32_t rows, int32_t cols);

__device__
double vecNormColMaj(double* vec, int32_t rows, int32_t cols);