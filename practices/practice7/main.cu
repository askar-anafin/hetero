
#include <iostream> // Include input/output stream library
#include <vector> // Include vector container library
#include <numeric> // Include numeric algorithms library
#include <algorithm> // Include general algorithms library
#include <random> // Include random number generation library
#include <cuda_runtime.h> // Include CUDA runtime library
#include "kernels.cuh" // Include our custom kernels header

#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } // Macro to check CUDA API call errors, print error message and exit if failed

const int BLOCK_SIZE = 256; // Define the block size (number of threads per block)

// CPU Implementations
long long cpu_reduce(const std::vector<int>& data) { // Function for CPU reduction (summation)
    long long sum = 0; // Initialize sum variable
    for (int x : data) sum += x; // Iterate over all elements and add to sum
    return sum; // Return the total sum
} // End function

void cpu_scan(const std::vector<int>& input, std::vector<int>& output) { // Function for CPU scan (prefix sum)
    if (input.empty()) return; // If input is empty, return immediately
    output[0] = input[0]; // The first element of output is same as input
    for (size_t i = 1; i < input.size(); ++i) { // Loop from 2nd element to end
        output[i] = output[i - 1] + input[i]; // Current output is previous output plus current input
    } // End for
} // End function

// Helper: recursive scan function for large arrays
void gpu_scan_recursive(int* d_input, int* d_output, int n) { // Recursive function for GPU scan
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; // Calculate number of blocks required covering n elements
    
    // 1. Allocate memory for block sums
    int* d_block_sums; // Pointer for block sums array on device
    CHECK_CUDA(cudaMalloc(&d_block_sums, num_blocks * sizeof(int))); // Allocate device memory for block sums

    // 2. Scan within each block and collect sums
    scan_block_kernel<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_input, d_output, d_block_sums, n); // Launch kernel to scan blocks
    CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors

    // 3. If multiple blocks, scan the block sums recursively
    if (num_blocks > 1) { // Checks if there is more than one block
        int* d_block_sums_scanned; // Pointer for scanned block sums
        CHECK_CUDA(cudaMalloc(&d_block_sums_scanned, num_blocks * sizeof(int))); // Allocate memory for scanned block sums
        
        gpu_scan_recursive(d_block_sums, d_block_sums_scanned, num_blocks); // Recursively call this function to scan the block sums array
        
        // 4. Add the scanned block sums to the original blocks
        add_block_sum_kernel<<<num_blocks, BLOCK_SIZE>>>(d_output, d_block_sums_scanned, n); // Launch kernel to add base values to blocks
        CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors
        
        CHECK_CUDA(cudaFree(d_block_sums_scanned)); // Free the temporary array for scanned block sums
    } // End if
    
    CHECK_CUDA(cudaFree(d_block_sums)); // Free the block sums array
} // End function


void run_tests(int n) { // Function to run tests for a specific array size N
    std::cout << "\n--- Testing with N = " << n << " ---" << std::endl; // Print header for current test size

    // Host Data
    std::vector<int> h_data(n); // Create host vector of size n
    std::generate(h_data.begin(), h_data.end(), []() { return rand() % 10; }); // Fill vector with random numbers 0-9

    // Device Data
    int* d_input; // Pointer for input array on device
    int* d_output_reduce; // (Unused in current flow, declare kept for consistency if needed later)
    int* d_output_scan; // Pointer for scan output array on device
    
    CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(int))); // Allocate memory for input on device
    CHECK_CUDA(cudaMalloc(&d_output_scan, n * sizeof(int))); // Allocate memory for scan output on device
    CHECK_CUDA(cudaMemcpy(d_input, h_data.data(), n * sizeof(int), cudaMemcpyHostToDevice)); // Copy data from host to device

    // --- REDUCTION TEST (SHARED) ---
    // 2-pass reduction for simplicity/performance
    // Pass 1: Blocks -> Partial Sums
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; // Calculate grid size
    int* d_partial_sums; // Pointer for partial sums array
    CHECK_CUDA(cudaMalloc(&d_partial_sums, num_blocks * sizeof(int))); // Allocate memory for partial sums
    
    cudaEvent_t start, stop; // Declare CUDA events for timing
    cudaEventCreate(&start); // Create start event
    cudaEventCreate(&stop); // Create stop event

    cudaEventRecord(start); // Record start event
    
    // Kernel 1
    // Launch reduction kernel with dynamic shared memory size
    reduce_shared<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_input, d_partial_sums, n); 
    CHECK_CUDA(cudaGetLastError()); // Check kernel errors
    cudaEventRecord(stop); // Stop timing the GPU part (plus overhead of copy back below)
    
    // Copy partials back to CPU to finish
    std::vector<int> h_partials(num_blocks); // Vector to hold partial sums on host
    CHECK_CUDA(cudaMemcpy(h_partials.data(), d_partial_sums, num_blocks * sizeof(int), cudaMemcpyDeviceToHost)); // Copy partial sums from device to host
    
    // Finish on CPU
    long long gpu_sum_val = 0; // Initialize final GPU sum
    for (int x : h_partials) gpu_sum_val += x; // Sum up partial results on CPU
    
    cudaEventSynchronize(stop); // Wait for GPU to finish
    float ms_reduce = 0; // Variable for time
    cudaEventElapsedTime(&ms_reduce, start, stop); // Calculate elapsed time in ms

    // Verify Reduction
    cudaEventRecord(start); // Start timing CPU reduction
    long long cpu_sum_val = cpu_reduce(h_data); // Run CPU reduction benchmark
    cudaEventRecord(stop); // Stop timing CPU
    cudaEventSynchronize(stop); // Sync
    float ms_cpu_reduce = 0; // Variable for CPU time
    cudaEventElapsedTime(&ms_cpu_reduce, start, stop); // Calculate elapsed time

    std::cout << "Reduction Time GPU: " << ms_reduce << " ms" << std::endl; // Print GPU time
    std::cout << "Reduction Time CPU: " << ms_cpu_reduce << " ms" << std::endl; // Print CPU time
    std::cout << "CPU Sum: " << cpu_sum_val << ", GPU Sum: " << gpu_sum_val << std::endl; // Print sums
    if (cpu_sum_val == gpu_sum_val) std::cout << "[PASSED] Reduction" << std::endl; // Check equality
    else std::cout << "[FAILED] Reduction" << std::endl; // Report failure

    CHECK_CUDA(cudaFree(d_partial_sums)); // Free partial sums memory

    // --- SCAN TEST (SHARED / RECURSIVE) ---
    cudaEventRecord(start); // Start timing GPU scan
    gpu_scan_recursive(d_input, d_output_scan, n); // Run recursive GPU scan wrapper
    cudaEventRecord(stop); // Stop timing GPU scan
    cudaEventSynchronize(stop); // Wait for GPU to finish
    float ms_scan = 0; // Variable for time
    cudaEventElapsedTime(&ms_scan, start, stop); // Calculate elapsed time

    std::vector<int> h_utils_scan(n); // Vector to hold scan results on host
    CHECK_CUDA(cudaMemcpy(h_utils_scan.data(), d_output_scan, n * sizeof(int), cudaMemcpyDeviceToHost)); // Copy results from device to host

    // Verify Scan
    std::vector<int> cpu_scan_res(n); // Vector for CPU scan results
    
    cudaEventRecord(start); // Start timing CPU scan
    cpu_scan(h_data, cpu_scan_res); // Run CPU scan benchmark
    cudaEventRecord(stop); // Stop timing CPU scan
    cudaEventSynchronize(stop); // Sync
    float ms_cpu_scan = 0; // Variable for CPU time
    cudaEventElapsedTime(&ms_cpu_scan, start, stop); // Calculate elapsed time
    
    bool scan_passed = true; // Flag for correctness
    for(int i = 0; i < n; ++i) { // Loop through all elements
        // Check for mismatch
        if (h_utils_scan[i] != cpu_scan_res[i]) { 
            std::cout << "Mismatch at " << i << ": CPU " << cpu_scan_res[i] << " vs GPU " << h_utils_scan[i] << std::endl; // Print error
            scan_passed = false; // Set flag false
            break; // Exit verification loop
        } // End if
    } // End for
    
    std::cout << "Scan Time GPU: " << ms_scan << " ms" << std::endl; // Print GPU time
    std::cout << "Scan Time CPU: " << ms_cpu_scan << " ms" << std::endl; // Print CPU time
    if (scan_passed) std::cout << "[PASSED] Scan" << std::endl; // Print Pass
    else std::cout << "[FAILED] Scan" << std::endl; // Print Fail


    CHECK_CUDA(cudaFree(d_input)); // Free input memory
    CHECK_CUDA(cudaFree(d_output_scan)); // Free output memory
} // End function

int main() { // Main function
    // 1024
    run_tests(1024); // Run tests for small array
    
    // 1M
    run_tests(1000000); // Run tests for 1 million elements
    
    // 10M
    run_tests(10000000); // Run tests for 10 million elements
    
    return 0; // Return success
} // End main
