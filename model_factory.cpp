#include "singleton.h"
#include "model_factory.h"


void ModelFactoryRegistry::RegisterModel(const std::string &name, int32_t dim, int32_t param_num, ModelType type, ModelFunction func,
                                         ModelFunctionVar func_var, ModelFunctionVar func_var_rev,
                                         ModelFunctionVar func_var_trans_rev, ModelFunctionQ func_Q, const std::string &model_param, 
										 const std::string &model_phase, const std::string &model_equat) {
    Storage &storage = Singleton<Storage>::Instance();
    ModelData data(name, dim, param_num, type, func, func_var, func_var_rev, func_var_trans_rev, func_Q, model_param, model_phase, model_equat);
    storage._data.insert(std::make_pair(name, std::move(data)));
}