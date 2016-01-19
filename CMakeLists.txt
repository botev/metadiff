cmake_minimum_required(VERSION 2.8) # Could be even lower. Works with 3.0
project(metadiff)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -Werror=return-type")

include_directories(include)
include_directories(/opt/eigen/)
include_directories(/opt/intel/mkl/include)
link_directories(/usr/local/lib /opt/intel/intel-opencl-1.2-5.0.0.43/opencl/lib64)

link_libraries(dl)
link_libraries(curl)

add_subdirectory(tests)
add_executable(mnist_hinton_e.o examples/mnist_hinton_e.cpp)
