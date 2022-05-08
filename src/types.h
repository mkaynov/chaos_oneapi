#pragma once

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
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

typedef void(*diffSysFunc)(const double*, double*, const double*);

typedef void(*diffSysFuncVar)(const double*, double*, const double*, const double*);

typedef void(*linearSysFunc)(double*, const double*, const double*);

//typedef Eigen::MatrixXd(*linMatOfSys)(const double*, const double*);