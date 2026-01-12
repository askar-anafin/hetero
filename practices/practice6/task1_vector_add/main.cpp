// Define the OpenCL target version to 3.0
#define CL_TARGET_OPENCL_VERSION 300
// Include the OpenCL header file
#include <CL/cl.h>
// Include the standard input/output stream library
#include <iostream>
// Include the vector container library
#include <vector>
// Include the string library
#include <string>
// Include the chrono library for time measurement
#include <chrono>
// Include the file stream library
#include <fstream>
// Include the string stream library
#include <sstream>

// MACRO to check for OpenCL errors
// Prints the error message and error code to standard error
// Exits the program with failure status
#define CHECK_ERROR(err, msg) \
    if (err != CL_SUCCESS) { \
        std::cerr << "Error: " << msg << " (" << err << ")" << std::endl; \
        exit(1); \
    }


// Define the number of elements as a constant (10 million)
const int N = 10000000; // 10 million elements
// Calculate the size of the data in bytes
const size_t DATA_SIZE = N * sizeof(float);

// Function to load the kernel source code from a file
std::string load_kernel_source(const char* filename) {
    // Open the file
    std::ifstream file(filename);
    // Check if the file opened successfully
    if (!file.is_open()) {
        // Print an error message if the file could not be opened
        std::cerr << "Failed to load kernel file: " << filename << std::endl;
        // Exit the program
        exit(1);
    }
    // Create a string stream to hold the file contents
    std::stringstream ss;
    // Read the file buffer into the string stream
    ss << file.rdbuf();
    // Return the string content of the stream
    return ss.str();
}

// Function to run the benchmark on a specific OpenCL device
void benchmark_device(cl_platform_id platform, cl_device_id device, const std::string& source) {
    // Variable to hold error codes
    cl_int err;

    // Buffer to hold the device name
    char deviceName[128];
    // Get the device name
    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, deviceName, NULL);
    // Variable to hold the device type
    cl_device_type deviceType;
    // Get the device type
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, NULL);
    
    // Determine the device type string (CPU, GPU, or Other)
    std::string typeStr = (deviceType & CL_DEVICE_TYPE_CPU) ? "CPU" : 
                          (deviceType & CL_DEVICE_TYPE_GPU) ? "GPU" : "Other";

    // Print the device type and name
    std::cout << "Running on " << typeStr << ": " << deviceName << std::endl;

    // Create an OpenCL context for the device
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    // Check for errors during context creation
    CHECK_ERROR(err, "clCreateContext");

    // Create a command queue for the device
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    // Check for errors during command queue creation
    CHECK_ERROR(err, "clCreateCommandQueue");

    // Get the pointer to the kernel source string
    const char* src_ptr = source.c_str();
    // Get the length of the kernel source string
    size_t src_len = source.length();
    // Create the program object from the source code
    cl_program program = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &err);
    // Check for errors during program creation
    CHECK_ERROR(err, "clCreateProgramWithSource");

    // Build the program for the device
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    // Check if the build failed
    if (err != CL_SUCCESS) {
        // Variable to hold the size of the build log
        size_t log_size;
        // Get the size of the build log
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        // Create a buffer to hold the build log
        std::vector<char> log(log_size);
        // Get the build log content
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        // Print the build log to standard error
        std::cerr << "Build Log:\n" << log.data() << std::endl;
        // Exit the program
        exit(1);
    }

    // Create the kernel object for the "vector_add" function
    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    // Check for errors during kernel creation
    CHECK_ERROR(err, "clCreateKernel");

    // Prepare Host Data
    // Create vector A with N elements initialized to 1.0f
    std::vector<float> h_A(N, 1.0f);
    // Create vector B with N elements initialized to 2.0f
    std::vector<float> h_B(N, 2.0f);
    // Create vector C with N elements to hold the result
    std::vector<float> h_C(N);

    // Create Device Buffers
    // Create buffer for A (Read Only, Copy from Host)
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, DATA_SIZE, h_A.data(), &err);
    // Check for errors
    CHECK_ERROR(err, "clCreateBuffer A");
    // Create buffer for B (Read Only, Copy from Host)
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, DATA_SIZE, h_B.data(), &err);
    // Check for errors
    CHECK_ERROR(err, "clCreateBuffer B");
    // Create buffer for C (Write Only)
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, DATA_SIZE, NULL, &err);
    // Check for errors
    CHECK_ERROR(err, "clCreateBuffer C");

    // Set Kernel Arguments
    // Set argument 0 to d_A
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    // Set argument 1 to d_B
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    // Set argument 2 to d_C
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    // Check for errors setting arguments
    CHECK_ERROR(err, "clSetKernelArg");

    // Set global work size to the number of elements
    size_t globalSize = N;
    
    // Warmup Run (to initialize device state)
    // Enqueue the kernel for execution
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
    // Wait for the command queue to finish
    clFinish(queue);

    // Measure Execution Time
    // Get the start time
    auto start = std::chrono::high_resolution_clock::now();
    
    // Enqueue the kernel for execution again (Timed Run)
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
    // Check for errors enqueuing the kernel
    CHECK_ERROR(err, "clEnqueueNDRangeKernel");
    // Wait for the command queue to finish execution
    clFinish(queue); 

    // Get the end time
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the duration in milliseconds
    std::chrono::duration<double, std::milli> duration = end - start;

    // Print the execution time
    std::cout << "Time: " << duration.count() << " ms" << std::endl;

    // Read Result from Device to Host
    // Enqueue a read command to copy data from d_C (device) to h_C (host)
    err = clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, DATA_SIZE, h_C.data(), 0, NULL, NULL);
    // Check for errors reading the buffer
    CHECK_ERROR(err, "clEnqueueReadBuffer");

    // Verify Results
    // Flag to track correctness
    bool correct = true;
    // Loop through all elements
    for (int i = 0; i < N; i++) {
        // Check if the result is correct (1.0 + 2.0 should be 3.0)
        if (h_C[i] != 3.0f) {
            // Print error message if incorrect
            std::cerr << "Verification failed at " << i << ": " << h_C[i] << " != 3.0" << std::endl;
            // Set flag to false
            correct = false;
            // Break the loop
            break;
        }
    }
    // Print success message if verify passed
    if (correct) std::cout << "Verification Passed!" << std::endl;

    // Cleanup Resources
    // Release buffer d_A
    clReleaseMemObject(d_A);
    // Release buffer d_B
    clReleaseMemObject(d_B);
    // Release buffer d_C
    clReleaseMemObject(d_C);
    // Release kernel
    clReleaseKernel(kernel);
    // Release program
    clReleaseProgram(program);
    // Release command queue
    clReleaseCommandQueue(queue);
    // Release context
    clReleaseContext(context);
    
    // Print separator line
    std::cout << "------------------------------------------------" << std::endl;
}

// Main function
int main() {
    // Variable to hold the number of platforms
    cl_uint numPlatforms;
    // Get the number of available OpenCL platforms
    clGetPlatformIDs(0, NULL, &numPlatforms);
    // Check if no platforms were found
    if (numPlatforms == 0) {
        // Print error message
        std::cerr << "No OpenCL platforms found." << std::endl;
        // Return with error code
        return 1;
    }

    // Create a vector to store platform IDs
    std::vector<cl_platform_id> platforms(numPlatforms);
    // Get the platform IDs
    clGetPlatformIDs(numPlatforms, platforms.data(), NULL);

    // Load the kernel source code from file
    std::string kernelSource = load_kernel_source("kernel.cl");

    // CPU Benchmark Section
    // Print message indicating CPU start
    std::cout << "Running CPU Control (Sequential)..." << std::endl;
    // Create vector A initialized with 1.0f
    std::vector<float> h_A(N, 1.0f);
    // Create vector B initialized with 2.0f
    std::vector<float> h_B(N, 2.0f);
    // Create vector C to store results
    std::vector<float> h_C(N);

    // Start CPU timer
    auto start_cpu = std::chrono::high_resolution_clock::now();
    // Perform sequential vector addition
    for (int i = 0; i < N; i++) {
        h_C[i] = h_A[i] + h_B[i];
    }
    // End CPU timer
    auto end_cpu = std::chrono::high_resolution_clock::now();
    // Calculate CPU duration
    std::chrono::duration<double, std::milli> duration_cpu = end_cpu - start_cpu;
    // Print CPU execution time
    std::cout << "CPU Time: " << duration_cpu.count() << " ms" << std::endl;
    // Print separator
    std::cout << "------------------------------------------------" << std::endl;

    // OpenCL Benchmark Section
    // Iterate over each platform
    for (auto platform : platforms) {
        // Variable to hold number of devices
        cl_uint numDevices;
        // Get number of devices for this platform
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
        // Continue if no devices found
        if (numDevices == 0) continue;

        // Create vector to store device IDs
        std::vector<cl_device_id> devices(numDevices);
        // Get device IDs
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), NULL);

        // Iterate over each device
        for (auto device : devices) {
            // Run benchmark for this device
            benchmark_device(platform, device, kernelSource);
        }
    }

    // Return success
    return 0;
}
