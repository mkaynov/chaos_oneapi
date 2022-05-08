#pragma once

#include <vector>
#include <functional>

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
#include "types.h"


enum class ModelType
{
	flow,
	map
};

class ModelFactoryRegistry
{
protected:

    using ModelFunction = std::function<void(diffSysFunc*)>;
    using ModelFunctionVar = std::function<void(diffSysFuncVar*)>;
	using ModelFunctionQ = std::function<void(diffSysFuncVar*)>;

    static void
    RegisterModel(const std::string &name, int32_t dim, int32_t param_num, ModelType type, ModelFunction func,
                  ModelFunctionVar func_var, ModelFunctionVar func_var_rev, ModelFunctionVar func_var_trans_rev, ModelFunctionQ func_Q, 
				  const std::string &model_param, const std::string &model_phase, const std::string &model_equat);

public:

    struct ModelData {
        ModelData(const std::string &name, int32_t dim, int32_t param_num, ModelType type, ModelFunction func,
                  ModelFunctionVar func_var, ModelFunctionVar func_var_rev, ModelFunctionVar func_var_trans_rev,
                  ModelFunctionQ func_Q, const std::string &model_param, const std::string &model_phase, const std::string &model_equat) :
                _dimension(dim), _param_number(param_num), _type(type), _model_name(name), _model_func(func),
                _func_var(func_var), _func_var_rev(func_var_rev),  _func_var_trans_rev(func_var_trans_rev), _func_Q(func_Q),
				_model_param(model_param), _model_phase(model_phase), _model_equat(model_equat) {}

        int32_t _dimension;
        int32_t _param_number;
        std::string _model_name;
        ModelType _type;
        ModelFunction _model_func;
        ModelFunctionVar _func_var;
        ModelFunctionVar _func_var_rev;
        ModelFunctionVar _func_var_trans_rev;
        ModelFunctionQ _func_Q;
		std::string _model_param;
		std::string _model_phase;
		std::string _model_equat;
    };

    struct Storage
    {
        std::map<std::string, ModelData> _data;
    };
};