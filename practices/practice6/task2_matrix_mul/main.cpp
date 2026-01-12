// Define the OpenCL target version to 3.0
#define CL_TARGET_OPENCL_VERSION 300
// Include OpenCL header
#include <CL/cl.h>
// Include Input/Output Stream
#include <iostream>
// Include Vector container
#include <vector>
// Include String library
#include <string>
// Include Chrono for timing
#include <chrono>
// Include File Stream
#include <fstream>
// Include String Stream
#include <sstream>
// Include Math library
#include <cmath>

// Macro to check OpenCL error codes
// Print error message to standard error
// Exit application with error
#define CHECK_ERROR(err, msg) \
    if (err != CL_SUCCESS) { \
        std::cerr << "Error: " << msg << " (" << err << ")" << std::endl; \
        exit(1); \
    }


// Define Constant Width for Matrix (1024 x 1024)
const int WIDTH = 1024; // Matrix size 1024x1024
// Calculate total data size in bytes
const size_t DATA_SIZE = WIDTH * WIDTH * sizeof(float);

// Function to load kernel source from file
std::string load_kernel_source(const char* filename) {
    // Open file stream
    std::ifstream file(filename);
    // Check if file is open
    if (!file.is_open()) {
        // Print error if file not found
        std::cerr << "Failed to load kernel file: " << filename << std::endl;
        // Exit application
        exit(1);
    }
    // Create string stream
    std::stringstream ss;
    // Read file buffer into string stream
    ss << file.rdbuf();
    // Return string content
    return ss.str();
}

// Function to benchmark a specific OpenCL device
void benchmark_device(cl_platform_id platform, cl_device_id device, const std::string& source) {
    // Error code variable
    cl_int err;

    // Buffer for device name
    char deviceName[128];
    // Get Device Name
    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, deviceName, NULL);
    // Variable for device type
    cl_device_type deviceType;
    // Get Device Type
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, NULL);
    
    // Determine type string
    std::string typeStr = (deviceType & CL_DEVICE_TYPE_CPU) ? "CPU" : 
                          (deviceType & CL_DEVICE_TYPE_GPU) ? "GPU" : "Other";

    // Print running message
    std::cout << "Running on " << typeStr << ": " << deviceName << std::endl;

    // Create OpenCL Context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    // Check error
    CHECK_ERROR(err, "clCreateContext");

    // Create Command Queue
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    // Check error
    CHECK_ERROR(err, "clCreateCommandQueue");

    // Build Program from Source
    const char* src_ptr = source.c_str();
    size_t src_len = source.length();
    cl_program program = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &err);
    // Check error
    CHECK_ERROR(err, "clCreateProgramWithSource");

    // Compile Program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    // Check if compilation failed
    if (err != CL_SUCCESS) {
        // Get build log size
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        // Allocate buffer for log
        std::vector<char> log(log_size);
        // Get build log
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        // Print build log
        std::cerr << "Build Log:\n" << log.data() << std::endl;
        // Exit
        exit(1);
    }

    // Create Kernel object
    cl_kernel kernel = clCreateKernel(program, "matrix_mul", &err);
    // Check error
    CHECK_ERROR(err, "clCreateKernel");

    // Prepare Host Data
    // Create matrix A
    std::vector<float> h_A(WIDTH * WIDTH);
    // Create matrix B
    std::vector<float> h_B(WIDTH * WIDTH);
    // Create matrix C for results
    std::vector<float> h_C(WIDTH * WIDTH);

    // Initialize matrices
    for(int i=0; i<WIDTH*WIDTH; ++i) {
        h_A[i] = 1.0f; // Fill A with 1.0
        h_B[i] = 2.0f; // Fill B with 2.0
    }

    // Create Device Buffers
    // Buffer A (Read Only)
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, DATA_SIZE, h_A.data(), &err);
    CHECK_ERROR(err, "clCreateBuffer A");
    // Buffer B (Read Only)
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, DATA_SIZE, h_B.data(), &err);
    CHECK_ERROR(err, "clCreateBuffer B");
    // Buffer C (Write Only)
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, DATA_SIZE, NULL, &err);
    CHECK_ERROR(err, "clCreateBuffer C");

    // Set Argument 0 (A)
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    // Set Argument 1 (B)
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    // Set Argument 2 (C)
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    // Set Argument 3 (Width)
    int width = WIDTH;
    err |= clSetKernelArg(kernel, 3, sizeof(int), &width);
    // Check error
    CHECK_ERROR(err, "clSetKernelArg");

    // Define Global Work Size (2D: WIDTH x WIDTH)
    size_t globalSize[2] = {static_cast<size_t>(WIDTH), static_cast<size_t>(WIDTH)};
    
    // Start Timer
    auto start = std::chrono::high_resolution_clock::now();
    
    // Enqueue Kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    CHECK_ERROR(err, "clEnqueueNDRangeKernel");
    // Wait for completion
    clFinish(queue); 

    // Stop Timer
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate Duration
    std::chrono::duration<double, std::milli> duration = end - start;

    // Print Time
    std::cout << "Time: " << duration.count() << " ms" << std::endl;

    // Read Result Buffer
    err = clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, DATA_SIZE, h_C.data(), 0, NULL, NULL);
    CHECK_ERROR(err, "clEnqueueReadBuffer");

    // Verify Results
    // Expected value: Sum(A[i]*B[i]) -> Sum(1.0 * 2.0) repeated WIDTH times -> 2.0 * WIDTH
    float expected = 2.0f * WIDTH;
    bool correct = true;
    // Iterate over all elements
    for (int i = 0; i < WIDTH * WIDTH; i++) {
        // Check difference with tolerance
        if (std::abs(h_C[i] - expected) > 0.001f) {
            // Print failure
            std::cerr << "Verification failed at " << i << ": " << h_C[i] << " != " << expected << std::endl;
            correct = false;
            break; 
        }
    }
    // Print success
    if (correct) std::cout << "Verification Passed!" << std::endl;

    // Cleanup Resources
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    // Print Separator
    std::cout << "------------------------------------------------" << std::endl;
}

// Main Function
int main() {
    // Number of platforms
    cl_uint numPlatforms;
    // Get Platform Count
    clGetPlatformIDs(0, NULL, &numPlatforms);
    // Check if platforms exist
    if (numPlatforms == 0) {
        std::cerr << "No OpenCL platforms found." << std::endl;
        return 1;
    }

    // Get Platform IDs
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), NULL);

    // Load Kernel Source
    std::string kernelSource = load_kernel_source("matrix_kernel.cl");

    // Run CPU Benchmark (Sequential)
    std::cout << "Running CPU Control (Sequential)..." << std::endl;
    // Initialize Data for CPU
    std::vector<float> h_A(WIDTH * WIDTH, 1.0f);
    std::vector<float> h_B(WIDTH * WIDTH, 2.0f);
    std::vector<float> h_C(WIDTH * WIDTH);

    // Start CPU Timer
    auto start_cpu = std::chrono::high_resolution_clock::now();
    
    // Perform Naive Matrix Multiplication iteratively
    for (int i = 0; i < WIDTH; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < WIDTH; ++k) {
                sum += h_A[i * WIDTH + k] * h_B[k * WIDTH + j];
            }
            h_C[i * WIDTH + j] = sum;
        }
    }

    // Stop CPU Timer
    auto end_cpu = std::chrono::high_resolution_clock::now();
    // Calculate Duration
    std::chrono::duration<double, std::milli> duration_cpu = end_cpu - start_cpu;
    // Print CPU Time
    std::cout << "CPU Time: " << duration_cpu.count() << " ms" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // Iterate over OpenCL Platforms
    for (auto platform : platforms) {
        // Device Count
        cl_uint numDevices;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
        if (numDevices == 0) continue;

        // Get Device IDs
        std::vector<cl_device_id> devices(numDevices);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), NULL);

        // Iterate over Devices
        for (auto device : devices) {
            // Run Benchmark
            benchmark_device(platform, device, kernelSource);
        }
    }

    // Return Success
    return 0;
}
