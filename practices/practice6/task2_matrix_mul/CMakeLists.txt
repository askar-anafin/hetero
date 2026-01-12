add_executable(matrix_mul main.cpp)
target_link_libraries(matrix_mul OpenCL::OpenCL)
# Ensure kernel.cl is copied to the build directory
configure_file(matrix_kernel.cl ${CMAKE_CURRENT_BINARY_DIR}/matrix_kernel.cl COPYONLY)
