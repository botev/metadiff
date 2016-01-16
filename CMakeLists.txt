cmake_minimum_required(VERSION 2.8) # Could be even lower. Works with 3.0
project(metadiff)

FIND_PACKAGE(ArrayFire REQUIRED PATHS /opt/arrayfire-3/share/ArrayFire/cmake/)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -Werror=return-type")

include_directories(include)
include_directories(${ArrayFire_INCLUDE_DIRS})
link_directories(/usr/local/lib /opt/intel/intel-opencl-1.2-5.0.0.43/opencl/lib64)

link_libraries(${ArrayFire_Unified_LIBRARIES})
link_libraries(dl)
#link_libraries(OpenCL)
link_libraries(curl)

add_subdirectory(tests)
add_executable(mnist_hinton.o examples/mnist_hinton.cpp)
