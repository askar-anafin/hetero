
#ifndef KERNELS_CUH // Check if KERNELS_CUH is not defined to prevent double inclusion
#define KERNELS_CUH // Define KERNELS_CUH symbol

#include <cuda_runtime.h> // Include CUDA runtime API headers
#include <device_launch_parameters.h> // Include CUDA device launch parameters (blockIdx, etc.)
#include <stdio.h> // Include standard I/O for printf (used for debugging if needed)

// --- Reduction Kernels ---

// Naive atomic reduction (Global Memory)
// Each thread adds its value to a global sum. Very slow due to contention.
__global__ void reduce_global_atomic(const int* input, int* output, int n) { // Kernel definition for atomic reduction
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global thread ID
    if (tid < n) { // Check if thread ID is within array bounds
        atomicAdd(output, input[tid]); // Atomically add current element to output sum
    } // End if
} // End kernel

// Optimized Reduction using Shared Memory
// Performs tree-based reduction within a block.
// Each block writes its partial sum to 'output'.
__global__ void reduce_shared(const int* input, int* output, int n) { // Kernel definition for shared memory reduction
    extern __shared__ int sdata[]; // Declare dynamic shared memory array

    unsigned int tid = threadIdx.x; // Get thread ID within the block
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; // Get global thread ID

    // Load input into shared memory (with bounds check)
    // Optimization: Perform first add during load can reduce threads by half, 
    // but here we stick to basic load for clarity.
    sdata[tid] = (i < n) ? input[i] : 0; // Load input element to shared memory if in bounds, else load 0
    __syncthreads(); // Synchronize all threads in the block to ensure loading is done

    // Do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) { // Loop for reduction steps: stride s starts at half block size, divides by 2 each step
        if (tid < s) { // Only the first 's' threads work in this step
            sdata[tid] += sdata[tid + s]; // Add element from upper half to lower half
        } // End if
        __syncthreads(); // Synchronize threads before the next step to ensure writes are visible
    } // End for

    // Write result for this block to global mem
    if (tid == 0) { // Only thread 0 writes the result
        output[blockIdx.x] = sdata[0]; // Write the block's sum (at index 0) to global output array at block index
    } // End if
} // End kernel


// --- Scan (Prefix Sum) Kernels ---

// Naive Hillis-Steele Scan (Global Memory - inefficient for large arrays due to multiple passes/synchs)
// BUT, implementing a true multi-block scan is complex (Blelloch 'Scan-Then-Fan').
// We will implement the "Local Scan + Block Sum" approach steps.

// 1. Block Scan & Sum:
//    - Scans the block's data into 'output'
//    - Stores the total sum of the block into 'block_sums'
//    Uses Hillis-Steele for intra-block scan.
__global__ void scan_block_kernel(const int* input, int* output, int* block_sums, int n) { // Kernel for block-level scan
    extern __shared__ int temp[]; // Declare dynamic shared memory array for scan operations

    int tid = threadIdx.x; // Get thread ID within the block
    int abs_idx = blockIdx.x * blockDim.x + tid; // Get global thread ID (absolute index)
    
    // Load to shared
    if (abs_idx < n) temp[tid] = input[abs_idx]; // Load global input to shared memory if in bounds
    else temp[tid] = 0; // Load 0 (identity for sum) if out of bounds
    __syncthreads(); // Synchronize threads after loading

    // Hillis-Steele Scan within shared memory
    // (Simpler to implement than Blelloch avoiding bank conflicts for now)
    for (int stride = 1; stride < blockDim.x; stride *= 2) { // Loop for scan steps: stride doubles each time
        int val = 0; // Temporary variable to store value to add
        if (tid >= stride) val = temp[tid - stride]; // If thread index >= stride, read value from 'stride' positions back
        __syncthreads(); // Synchronize to ensure all reads are done before writing
        if (tid >= stride) temp[tid] += val; // Add the read value to current position
        __syncthreads(); // Synchronize to ensure all writes are done before next step's reads
    } // End for

    // Write result to output
    if (abs_idx < n) output[abs_idx] = temp[tid]; // Write scanned value from shared memory to global output

    // Write block sum (last element) to auxiliary array
    if (tid == blockDim.x - 1) { // If this is the last thread in the block
        if (block_sums != nullptr) // Check if block_sums pointer is valid (not null)
            block_sums[blockIdx.x] = temp[tid]; // Write total sum of this block to block_sums array
    } // End if
} // End kernel

// 2. Uniform Add:
//    - Adds a value (from the scanned block sums) to all elements of a block.
__global__ void add_block_sum_kernel(int* output, const int* scanned_block_sums, int n) { // Kernel to add block increments
    int tid = threadIdx.x; // Get thread ID within the block
    int block_id = blockIdx.x; // Get block ID
    int abs_idx = block_id * blockDim.x + tid; // Get global thread ID

    // Skip first block (nothing to add)
    if (block_id == 0) return; // First block already has correct prefix sums, return

    if (abs_idx < n) { // Check bounds
        output[abs_idx] += scanned_block_sums[block_id - 1]; // Add sum of all previous blocks (from scanned_block_sums) to current element
    } // End if
} // End kernel

#endif // End include guard
