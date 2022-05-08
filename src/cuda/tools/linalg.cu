#include "linalg.cuh"

__device__
double dotProduct(double* firstVec, double* secondVec, int32_t length) {
	double res = 0;
	for (int32_t i = 0; i < length; i++)
		res += firstVec[i] * secondVec[i];
	return res;
}

__device__
double vecNorm(double* vec, int32_t length) {
	double norm = 0;
	for (int32_t i = 0; i < length; i++)
		norm += vec[i] * vec[i];
	return sqrt(norm);
}

__device__
void normalizeVecs(double* vecs, int32_t dimension, int32_t numVec, double eps) {
	double norm = 0;
	for (int32_t i = 0; i < numVec; i++) {
		norm = vecNorm(&vecs[i * dimension], dimension);
		for (int32_t j = 0; j < dimension; j++)
			vecs[i * dimension + j] = (vecs[i * dimension + j] / norm) * eps;
	}
}

__device__
double getAngle(double* vec1, double* vec2, double dimension) {
    return acos((dotProduct(vec1, vec2, dimension)) / (vecNorm(vec1, dimension) * vecNorm(vec2, dimension)));
}

__device__
void ortVecs(double* vecs, int32_t dimension, int32_t numVecs, double* projSum) {
	double projCoeff = 0;

	for (int32_t i = 1; i < numVecs; i++) {
		for (int32_t j = 0; j < i; j++) {
			projCoeff = dotProduct(&vecs[j * dimension], &vecs[i * dimension], dimension) / dotProduct(&vecs[j * dimension], &vecs[j * dimension], dimension);
			for (int32_t k = 0; k < dimension; k++)
				projSum[k] += projCoeff * vecs[j * dimension + k];
		}
		for (int32_t j = 0; j < dimension; j++) {
			vecs[i * dimension + j] -= projSum[j];
			projSum[j] = 0;
		}
	}
}

__device__
void matrixTranspose(double* matr, int32_t rows, int32_t cols) {
	double temp;

	for (int32_t i = 0; i < rows; i++)
		for (int32_t j = 0; j < cols; j++) {
			if (j > i) {
				temp = matr[i * cols + j];
				matr[i * cols + j] = matr[j * cols + i];
				matr[j * cols + i] = temp;
			}
		}
}

__device__
void matrixMult(double* res, double* A, int32_t rowsA, int32_t colsA, double* B, int32_t rowsB, int32_t colsB) {
	for (int32_t i = 0; i < rowsA; i++)
		for (int32_t j = 0; j < colsB; j++) {
			res[i * colsB + j] = 0;
			for (int32_t k = 0; k < colsA; k++)
				res[i * colsB + j] += (A[i * colsA + k] * B[k * colsB + j]);
		}
}

__device__
double dotProductColMaj(double* firstVec, double* secondVec, int32_t rows, int32_t cols) {
	double res = 0;
	for (int32_t i = 0; i < rows; i++)
		res += firstVec[i * cols] * secondVec[i * cols];
	return res;
}

__device__
double vecNormColMaj(double* vec, int32_t rows, int32_t cols) {
	double norm = 0;
	for (int32_t i = 0; i < rows; i++)
		norm += vec[i * cols] * vec[i * cols];
	return sqrt(norm);
}

__device__
void normalizeVecsColMaj(double* vecs, int32_t rows, int32_t cols, double eps) {
	double norm = 0;
	for (int32_t i = 0; i < rows; i++) {
		norm = vecNormColMaj(&vecs[i], rows, cols);
		for (int32_t j = 0; j < cols; j++)
			vecs[i + j * cols] = (vecs[i + j * cols] / norm) * eps;
	}
}

__device__
void ortVecsColMaj(double* vecs, int32_t rows, int32_t cols, double* projSum) {
	double projCoeff = 0;

	for (int32_t i = 1; i < cols; i++) {
		for (int32_t j = 0; j < i; j++) {
			projCoeff = dotProductColMaj(&vecs[j], &vecs[i], rows, cols) / dotProductColMaj(&vecs[j], &vecs[j], rows, cols);
			for (int32_t k = 0; k < rows; k++)
				projSum[k] += projCoeff * vecs[j + k * cols];
		}
		for (int32_t j = 0; j < rows; j++) {
			vecs[i + j * cols] -= projSum[j];
			projSum[j] = 0;
		}
	}
}

__device__
void qrDecomposition(double* R, double* Q, double* A, int32_t rows, int32_t cols, double* projSum) {
	memcpy(Q, A, rows * cols * sizeof(double));
	ortVecsColMaj(Q, rows, cols, projSum);
	normalizeVecsColMaj(Q, rows, cols, 1);
	matrixTranspose(Q, cols, rows);
	matrixMult(R, Q, rows, cols, A, rows, cols);
	matrixTranspose(Q, cols, rows);
}