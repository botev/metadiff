include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})


set(DGTEST_HAS_PTHREAD 1)
add_executable(runSymbolicTests monomials.cpp)
target_link_libraries(runSymbolicTests gtest gtest_main)
