#pragma once

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <cmath>
//#include <math.h>
//#include "Dense"
//#include "Eigenvalues"
#include <cstdlib>
#include <iostream>
#include <chrono>

#pragma omp declare simd
typedef void(*diffSysFunc)(const double*, double*, const double*);

#pragma omp declare simd
typedef void(*diffSysFuncVar)(const double*, double*, const double*, const double*);

#pragma omp declare simd
typedef void(*linearSysFunc)(double*, const double*, const double*);

//typedef Eigen::MatrixXd(*linMatOfSys)(const double*, const double*);