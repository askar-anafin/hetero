cmake_minimum_required(VERSION 3.10)
project(practice6_opencl)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

add_subdirectory(task1_vector_add)
add_subdirectory(task2_matrix_mul)
