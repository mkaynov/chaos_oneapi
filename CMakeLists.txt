cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(chaos LANGUAGES CXX CUDA)
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

file(GLOB CUDA_SOURCE_FILES
        src/cuda/models/flows/*.cu
        src/cuda/models/maps/*.cu
        )
file(GLOB CUDA_HEADER_FILES
        src/cuda/models/flows/*.cuh
        src/cuda/models/maps/*.cuh
        )
file(GLOB CUDA_SYMBDYN_FILES
        src/cuda/*.cu
        src/cuda/tools/*.cu
        )
add_library(models STATIC
        ${CUDA_SOURCE_FILES}
        ${CUDA_HEADER_FILES}
        ${CUDA_SYMBDYN_FILES}
        src/cuda/utils.cuh)

target_compile_features(models PUBLIC cxx_std_11)
target_include_directories(models PRIVATE src)
set_target_properties(models PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(SRC
        src/model_factory.cpp
        src/tool_params.cpp
        src/fileHandler.cpp
        src/TaskLyapunov.cpp
        )

file(GLOB HEADERS
        src/*.h
        )

add_executable(chaos ${SRC} ${HEADERS} src/main.cpp)
target_include_directories(chaos PRIVATE src)
target_link_libraries(chaos PRIVATE models)
target_compile_features(chaos PUBLIC cxx_std_11)
set_target_properties(chaos PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#### TESTS ###
#add_executable(tests ${SRC}
#        ./tests/unit/test_lyapunov.cpp
#        ./tests/unit/test_angle.cpp
#        ./tests/unit/test_kneadings.cpp
#        ./tests/catch_amalgamated.cpp
#        )
#target_include_directories(tests PRIVATE third-party/include/eigen third-party/include/json src tests)
#target_link_libraries(tests PRIVATE models)
#target_compile_features(tests PUBLIC cxx_std_11)
#set_target_properties(tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#### TESTS ###
#
#### PERF TESTS ###
#file(GLOB PERF_SRC ./tests/perfomance/*.cpp)
#add_executable(test_perf_hse ${SRC} ${PERF_SRC} ./tests/catch_amalgamated.cpp tests/perfomance/perf_utils.h)
#target_include_directories(test_perf_hse PRIVATE third-party/include/eigen third-party/include/json src tests)
#target_link_libraries(test_perf_hse PRIVATE models)
#target_compile_features(test_perf_hse PUBLIC cxx_std_11)
#set_target_properties(test_perf_hse PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#### PERF TESTS ###
