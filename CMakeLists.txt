cmake_minimum_required(VERSION 3.1)
project(metadiff)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -Werror=return-type")

include_directories(include)
include_directories(/opt/arrayfire-3/include)
link_directories(/opt/arrayfire-3/lib /usr/local/lib /opt/intel/intel-opencl-1.2-5.0.0.43/opencl/lib64)
link_libraries(af)
link_libraries(dl)
link_libraries(OpenCL)

add_subdirectory(tests)
add_executable(mnist_hinton examples/mnist_hinton.cpp)