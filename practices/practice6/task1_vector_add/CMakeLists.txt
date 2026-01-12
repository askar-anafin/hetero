add_executable(vector_add main.cpp)
target_link_libraries(vector_add OpenCL::OpenCL)
# Ensure kernel.cl is copied to the build directory
configure_file(kernel.cl ${CMAKE_CURRENT_BINARY_DIR}/kernel.cl COPYONLY)
