//File was auto genereted
//
//x= y
//y= -alpha*y - x*z + x
//z= -lamda*z + x**2.0

#include "shimizu.cuh"
#include "model_factory.h"
#include <cmath>


__device__
void shimizu(const double* state, double* res, const double* params) {
    res[0] = state[1] ;
    res[1] = - params[0] * state[1] - state[0] * state[2] + state[0] ;
    res[2] = - params[1] * state[2] + pow( state[0] , 2.0 ) ;
}

__device__
void shimizu_var(const double* state, double* res, const double* params, const double* stateOld) {
   res[0] = state[1] *  1  ;
   res[1] = state[0] * ( 1 - stateOld[2] ) + state[1] * ( - params[0] ) + state[2] * ( - stateOld[0] ) ;
   res[2] = state[0] * ( 2.0 * pow( stateOld[0] , 1.0 ) ) + state[2] * ( - params[1] ) ;
}

__device__
void shimizu_var_rev(const double* state, double* res, const double* params, const double* stateOld) {
   res[0] = -1 * ( state[0] * ( 0 ) +state[1] * ( 1 ) +state[2] * ( 0 )  );
   res[1] = -1 * ( state[0] * ( 1 - stateOld[2] ) +state[1] * ( - params[0] ) +state[2] * ( - stateOld[0] )  );
   res[2] = -1 * ( state[0] * ( 2.0 * pow( stateOld[0] , 1.0 ) ) +state[1] * ( 0 ) +state[2] * ( - params[1] )  );
}


__device__
void shimizuLinear_Q(const double* Q, double* res, const double* params, const double* state) {
    res[0] =  0  * Q[0] +  ( 1 ) * Q[3] +  0  * Q[6] ;
    res[1] =  0  * Q[1] +  ( 1 ) * Q[4] +  0  * Q[7] ;
    res[2] =  0  * Q[2] +  ( 1 ) * Q[5] +  0  * Q[8] ;
    res[3] =  ( 1 - state[2] ) * Q[0] +  ( - params[0] ) * Q[3] +  ( - state[0] ) * Q[6] ;
    res[4] =  ( 1 - state[2] ) * Q[1] +  ( - params[0] ) * Q[4] +  ( - state[0] ) * Q[7] ;
    res[5] =  ( 1 - state[2] ) * Q[2] +  ( - params[0] ) * Q[5] +  ( - state[0] ) * Q[8] ;
    res[6] =  ( 2.0 * pow( state[0] , 1.0 ) ) * Q[0] +  0  * Q[3] +  ( - params[1] ) * Q[6] ;
    res[7] =  ( 2.0 * pow( state[0] , 1.0 ) ) * Q[1] +  0  * Q[4] +  ( - params[1] ) * Q[7] ;
    res[8] =  ( 2.0 * pow( state[0] , 1.0 ) ) * Q[2] +  0  * Q[5] +  ( - params[1] ) * Q[8] ;
}


__device__
void shimizu_var_trans_rev(const double* state, double* res, const double* params, const double* stateOld) {
   res[0] = -1 * (state[1] * ( 1 - stateOld[2] ) + state[2] * ( 2.0 * pow( stateOld[0] , 1.0 ) ) );
   res[1] = -1 * (state[0] *  1  + state[1] * ( - params[0] ) );
   res[2] = -1 * (state[1] * ( - stateOld[0] ) + state[2] * ( - params[1] ) );
}


__device__ diffSysFunc d_shimizu = shimizu;
__device__ diffSysFuncVar d_shimizu_var = shimizu_var;
__device__ diffSysFuncVar d_shimizu_Q = shimizuLinear_Q;
__device__ diffSysFuncVar d_shimizu_var_rev = shimizu_var_rev;
__device__ diffSysFuncVar d_shimizu_var_trans_rev = shimizu_var_trans_rev;


static struct shimizuModelFactoryRegistry : public ModelFactoryRegistry
{
   shimizuModelFactoryRegistry()
   {
       auto func = [](diffSysFunc* p_func){
           cudaError_t err = cudaMemcpyFromSymbol(&(*p_func), d_shimizu, sizeof(diffSysFunc));
           if (err != cudaError_t::cudaSuccess)
           {
               std::cout << "Failed to copy model diff function on the device. Error: " << err << std::endl;
               exit(1);
           }
       };

       auto func_var = [](diffSysFuncVar* p_func){
           cudaError_t err = cudaMemcpyFromSymbol(&(*p_func), d_shimizu_var, sizeof(diffSysFuncVar));
           if (err != cudaError_t::cudaSuccess)
           {
               std::cout << "Failed to copy model diff function on the device. Error: " << err << std::endl;
               exit(1);
           }
       };

       auto func_Q = [](diffSysFuncVar* p_func){
           cudaError_t err = cudaMemcpyFromSymbol(&(*p_func), d_shimizu_Q, sizeof(diffSysFuncVar));
           if (err != cudaError_t::cudaSuccess)
           {
               std::cout << "Failed to copy model diff function on the device. Error: " << err << std::endl;
               exit(1);
           }
       };

       auto func_var_rev = [](diffSysFuncVar* p_func){
           cudaError_t err = cudaMemcpyFromSymbol(&(*p_func), d_shimizu_var_rev, sizeof(diffSysFuncVar));
           if (err != cudaError_t::cudaSuccess)
           {
               std::cout << "Failed to copy model diff function on the device. Error: " << err << std::endl;
               exit(1);
           }
       };

       auto func_var_trans_rev = [](diffSysFuncVar* p_func){
           cudaError_t err = cudaMemcpyFromSymbol(&(*p_func), d_shimizu_var_trans_rev, sizeof(diffSysFuncVar));
           if (err != cudaError_t::cudaSuccess)
           {
               std::cout << "Failed to copy model diff function on the device. Error: " << err << std::endl;
               exit(1);
           }
       };
   
       ModelType type = ModelType::flow;   
       RegisterModel("shimizu", 3, 2, type, func, func_var, func_var_rev, func_var_trans_rev, func_Q, "alpha, lamda", "x, y, z", "x' = y;y' = - alpha * y - x * z + x;z' = - lamda * z + x ** 2.0");
   }
} factory;
