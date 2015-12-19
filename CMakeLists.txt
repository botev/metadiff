cmake_minimum_required(VERSION 3.1)
project(autodiff)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -Werror=return-type")
set(SOURCE_FILES src/main.cpp include/metadiff.h)

include_directories(include)
include_directories(/opt/arrayfire-3/include)
link_directories(/opt/arrayfire-3/lib /usr/local/lib)
link_libraries(afcpu)
link_libraries(dl)

add_subdirectory(tests)
add_executable(metadiff ${SOURCE_FILES})