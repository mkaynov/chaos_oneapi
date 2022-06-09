//File was auto genereted
//
//x= y
//y= -alpha*y - x*z + x
//z= -lamda*z + x**2.0

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "shimizu.dp.hpp"
#include "model_factory.h"
#include <cmath>



void shimizu(const double* state, double* res, const double* params) {
    res[0] = state[1] ;
    res[1] = - params[0] * state[1] - state[0] * state[2] + state[0] ;
    res[2] = -params[1] * state[2] + state[0] * state[0];
}


void shimizu_var(const double* state, double* res, const double* params, const double* stateOld) {
   res[0] = state[1] *  1  ;
   res[1] = state[0] * ( 1 - stateOld[2] ) + state[1] * ( - params[0] ) + state[2] * ( - stateOld[0] ) ;
   res[2] = state[0] * (2.0 * sycl::pow<double>(stateOld[0], 1.0)) +
            state[2] * (-params[1]);
}


void shimizu_var_rev(const double* state, double* res, const double* params, const double* stateOld) {
   res[0] = -1 * ( state[0] * ( 0 ) +state[1] * ( 1 ) +state[2] * ( 0 )  );
   res[1] = -1 * ( state[0] * ( 1 - stateOld[2] ) +state[1] * ( - params[0] ) +state[2] * ( - stateOld[0] )  );
   res[2] = -1 * (state[0] * (2.0 * sycl::pow<double>(stateOld[0], 1.0)) +
                  state[1] * (0) + state[2] * (-params[1]));
}



void shimizuLinear_Q(const double* Q, double* res, const double* params, const double* state) {
    res[0] =  0  * Q[0] +  ( 1 ) * Q[3] +  0  * Q[6] ;
    res[1] =  0  * Q[1] +  ( 1 ) * Q[4] +  0  * Q[7] ;
    res[2] =  0  * Q[2] +  ( 1 ) * Q[5] +  0  * Q[8] ;
    res[3] =  ( 1 - state[2] ) * Q[0] +  ( - params[0] ) * Q[3] +  ( - state[0] ) * Q[6] ;
    res[4] =  ( 1 - state[2] ) * Q[1] +  ( - params[0] ) * Q[4] +  ( - state[0] ) * Q[7] ;
    res[5] =  ( 1 - state[2] ) * Q[2] +  ( - params[0] ) * Q[5] +  ( - state[0] ) * Q[8] ;
    res[6] = (2.0 * sycl::pow<double>(state[0], 1.0)) * Q[0] + 0 * Q[3] +
             (-params[1]) * Q[6];
    res[7] = (2.0 * sycl::pow<double>(state[0], 1.0)) * Q[1] + 0 * Q[4] +
             (-params[1]) * Q[7];
    res[8] = (2.0 * sycl::pow<double>(state[0], 1.0)) * Q[2] + 0 * Q[5] +
             (-params[1]) * Q[8];
}



void shimizu_var_trans_rev(const double* state, double* res, const double* params, const double* stateOld) {
   res[0] = -1 * (state[1] * (1 - stateOld[2]) +
                  state[2] * (2.0 * sycl::pow<double>(stateOld[0], 1.0)));
   res[1] = -1 * (state[0] *  1  + state[1] * ( - params[0] ) );
   res[2] = -1 * (state[1] * ( - stateOld[0] ) + state[2] * ( - params[1] ) );
}

dpct::global_memory<diffSysFunc, 0> d_shimizu(shimizu);
dpct::global_memory<diffSysFuncVar, 0> d_shimizu_var(shimizu_var);
dpct::global_memory<diffSysFuncVar, 0> d_shimizu_Q(shimizuLinear_Q);
dpct::global_memory<diffSysFuncVar, 0> d_shimizu_var_rev(shimizu_var_rev);
dpct::global_memory<diffSysFuncVar, 0>
    d_shimizu_var_trans_rev(shimizu_var_trans_rev);

static struct shimizuModelFactoryRegistry : public ModelFactoryRegistry
{
   shimizuModelFactoryRegistry() try {
       auto func = [](diffSysFunc* p_func){
           /*
           DPCT1003:3: Migrated API does not return error code. (*, 0) is
           inserted. You may need to rewrite this code.
           */
           int err = (dpct::get_default_queue()
                          .memcpy(&(*p_func), d_shimizu.get_ptr(),
                                  sizeof(diffSysFunc))
                          .wait(),
                      0);
           /*
           DPCT1000:2: Error handling if-stmt was detected but could not be
           rewritten.
           */
           if (err != 0)
           {
               /*
               DPCT1001:1: The statement could not be removed.
               */
               std::cout << "Failed to copy model diff function on the device. "
                            "Error: "
                         << err << std::endl;
               exit(1);
           }
       };

       auto func_var = [](diffSysFuncVar* p_func){
           /*
           DPCT1003:6: Migrated API does not return error code. (*, 0) is
           inserted. You may need to rewrite this code.
           */
           int err = (dpct::get_default_queue()
                          .memcpy(&(*p_func), d_shimizu_var.get_ptr(),
                                  sizeof(diffSysFuncVar))
                          .wait(),
                      0);
           /*
           DPCT1000:5: Error handling if-stmt was detected but could not be
           rewritten.
           */
           if (err != 0)
           {
               /*
               DPCT1001:4: The statement could not be removed.
               */
               std::cout << "Failed to copy model diff function on the device. "
                            "Error: "
                         << err << std::endl;
               exit(1);
           }
       };

       auto func_Q = [](diffSysFuncVar* p_func){
           /*
           DPCT1003:9: Migrated API does not return error code. (*, 0) is
           inserted. You may need to rewrite this code.
           */
           int err = (dpct::get_default_queue()
                          .memcpy(&(*p_func), d_shimizu_Q.get_ptr(),
                                  sizeof(diffSysFuncVar))
                          .wait(),
                      0);
           /*
           DPCT1000:8: Error handling if-stmt was detected but could not be
           rewritten.
           */
           if (err != 0)
           {
               /*
               DPCT1001:7: The statement could not be removed.
               */
               std::cout << "Failed to copy model diff function on the device. "
                            "Error: "
                         << err << std::endl;
               exit(1);
           }
       };

       auto func_var_rev = [](diffSysFuncVar* p_func){
           /*
           DPCT1003:12: Migrated API does not return error code. (*, 0) is
           inserted. You may need to rewrite this code.
           */
           int err = (dpct::get_default_queue()
                          .memcpy(&(*p_func), d_shimizu_var_rev.get_ptr(),
                                  sizeof(diffSysFuncVar))
                          .wait(),
                      0);
           /*
           DPCT1000:11: Error handling if-stmt was detected but could not be
           rewritten.
           */
           if (err != 0)
           {
               /*
               DPCT1001:10: The statement could not be removed.
               */
               std::cout << "Failed to copy model diff function on the device. "
                            "Error: "
                         << err << std::endl;
               exit(1);
           }
       };

       auto func_var_trans_rev = [](diffSysFuncVar* p_func){
           /*
           DPCT1003:15: Migrated API does not return error code. (*, 0) is
           inserted. You may need to rewrite this code.
           */
           int err = (dpct::get_default_queue()
                          .memcpy(&(*p_func), d_shimizu_var_trans_rev.get_ptr(),
                                  sizeof(diffSysFuncVar))
                          .wait(),
                      0);
           /*
           DPCT1000:14: Error handling if-stmt was detected but could not be
           rewritten.
           */
           if (err != 0)
           {
               /*
               DPCT1001:13: The statement could not be removed.
               */
               std::cout << "Failed to copy model diff function on the device. "
                            "Error: "
                         << err << std::endl;
               exit(1);
           }
       };
   
       ModelType type = ModelType::flow;   
       RegisterModel("shimizu", 3, 2, type, func, func_var, func_var_rev, func_var_trans_rev, func_Q, "alpha, lamda", "x, y, z", "x' = y;y' = - alpha * y - x * z + x;z' = - lamda * z + x ** 2.0");
   }
   catch (sycl::exception const &exc) {
     std::cerr << exc.what() << "Exception caught at file:" << __FILE__
               << ", line:" << __LINE__ << std::endl;
     std::exit(1);
   }
} factory;
