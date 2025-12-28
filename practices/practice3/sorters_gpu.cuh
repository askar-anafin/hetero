#ifndef SORTERS_GPU_CUH // Include guard to prevent multiple inclusions
#define SORTERS_GPU_CUH // Definition of the include guard

#include <cuda_runtime.h> // Include CUDA runtime API
#include <device_launch_parameters.h> // Include CUDA device launch parameters
#include <stdio.h> // Include standard I/O library

// Helper for atomic add wrapper if needed, but atomicAdd is standard.

// =============================================================
// MERGE SORT (Bitonic Sort for Block, Rank-based Merge)
// =============================================================

// Basic swap
__device__ void gpu_swap(int& a, int& b) { // Device function to swap two integers
    int t = a; a = b; b = t; // Standard swap logic using temporary variable
}

// Bitonic Sort for a single block (Power of 2 size assumed primarily, handles non-pot partially)
__global__ void block_sort_kernel(int* data, int n) { // Kernel to sort small chunks of data within a block
    int tid = threadIdx.x; // Get the thread ID within the block

    
    // Each thread handles one element? Or block sorts a chunk?
    // Let's assume blockDim.x is half the chunk size, or we sort chunks of size blockDim.x.
    // For simplicity: Sort chunks of size `blockDim.x * 2`? 
    // Let's implement specific "Sort Chunks" approach.
    // Each block sorts a segment of the array.
    // Simplest: Parallel Bitonic Sort inside Shared Memory.
    
    extern __shared__ int s_data[]; // Declare dynamic shared memory array
    
    // Load data
    int segment_size = blockDim.x;  // Define the size of the segment handled by this block
    int segment_idx = blockIdx.x * segment_size; // Calculate the starting index for this block's segment
    
    if (segment_idx + tid < n) // Check if the global index is within bounds
        s_data[tid] = data[segment_idx + tid]; // Load data from global memory to shared memory
    else
        s_data[tid] = 2147483647; // INT_MAX padding for out-of-bound threads
        
    __syncthreads(); // Synchronize threads to ensure all data is loaded
    
    // Bitonic Sort in Shared Memory
    for (int k = 2; k <= segment_size; k <<= 1) { // Outer loop for bitonic merge stages
        for (int j = k >> 1; j > 0; j >>= 1) { // Inner loop for comparisons within a stage
            int ixj = tid ^ j; // Calculate the index of the partner element
            if (ixj > tid) { // Ensure only one thread performs the swap for a pair
                if ((tid & k) == 0) { // Determine direction of sort (ascending/descending)
                    if (s_data[tid] > s_data[ixj]) gpu_swap(s_data[tid], s_data[ixj]); // Swap if needed (ascending)
                } else {
                    if (s_data[tid] < s_data[ixj]) gpu_swap(s_data[tid], s_data[ixj]); // Swap if needed (descending)
                }
            }
            __syncthreads(); // Synchronize threads after each comparison step
        }
    }
    
    // Write back
    if (segment_idx + tid < n) // Check bounds again before writing back
        data[segment_idx + tid] = s_data[tid]; // Write sorted data from shared memory to global memory
}

// Device function for binary search (Rank)
__device__ int co_rank(int val, int* arr, int n) { // Device function to find the rank of a value in an array
    int l = 0; // Initialize left boundary
    int r = n; // Initialize right boundary
    while (l < r) { // Binary search loop
        int mid = (l + r) / 2; // Calculate middle index
        if (arr[mid] < val) { // If value at mid is less than target
            l = mid + 1; // Search in the right half
        } else {
            r = mid; // Search in the left half
        }
    }
    return l; // Return the rank (insertion point)
}

// Rank-based Merge Kernel
// Merges src[base...base+size/2] and src[base+size/2...base+size] -> dst[base...base+size]
__global__ void merge_rank_kernel(int* src, int* dst, int n, int width) { // Kernel for merging sorted segments using rank-based approach
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Get global thread ID
    if (tid >= n) return; // Return if thread ID is out of bounds

    // Determine which two segments we are in
    // width is the size of the sorted segments being merged (e.g., 256, 512...)
    // Two segments: [start, mid) and [mid, end)
    
    // Actually, "global" merge rank:
    // Every element finds its rank in the *other* list.
    
    int pair_idx = tid / (2 * width); // index of the pair of blocks
    int start = pair_idx * 2 * width; // Calculate start index of the first segment
    int mid = start + width; // Calculate start index of the second segment (midpoint)
    int end = min(start + 2 * width, n); // Calculate end index of the second segment
    
    if (mid >= n) { // No right partner, just copy
        dst[tid] = src[tid]; // Copy element directly to destination
        return; // Exit
    }
    
    int val = src[tid]; // specific value to find rank for
    int final_pos; // Variable to store final position
    
    if (tid < mid) { // If element is in Left array
        // Element is in Left array
        // Find rank in Right array [mid, end)

        // Search in src[mid ... end-1]
        // Manual binary search
        int low = mid; // Initialize low for binary search in right segment
        int high = end; // Initialize high for binary search
        while (low < high) { // Binary search loop
            int m = low + (high - low) / 2; // Calculate midpoint
            if (src[m] < val) low = m + 1; // If middle element is smaller, search right
            else high = m; // otherwise search left
        }
        int rank_in_right = low - mid; // Calculate rank in the right segment
        int rank_in_left = tid - start; // Calculate rank in the left segment (its own index relative to start)
        final_pos = start + rank_in_left + rank_in_right; // Calculate global final position
    } else { // If element is in Right array
        // Element is in Right array
        // Find rank in Left array [start, mid)
        int low = start; // Initialize low for binary search in left segment
        int high = mid; // Initialize high
        while (low < high) { // Binary search loop
            int m = low + (high - low) / 2; // Calculate midpoint
            if (src[m] <= val) low = m + 1; // Strict/Non-strict for stability/uniqueness
            else high = m; // Search left
        }
        int rank_in_left = low - start; // Calculate rank in left segment
        int rank_in_right = tid - mid; // Calculate rank in right segment
        final_pos = start + rank_in_left + rank_in_right; // Calculate global final position
    }
    
    dst[final_pos] = val; // Write value to its final sorted position
}


// =============================================================
// QUICK SORT (Using Dynamic Parallelism)
// =============================================================

// --- Optimized Quick Sort with Threshold ---

__global__ void quick_sort_partition_and_launch(int* data, int left, int right, int depth) { // Kernel for recursive Quick Sort
    if (left >= right) return; // Base case: If range is invalid or single element, return

    // Small Chunk Optimization: Use sequential sort for small chunks
    // This dramatically reduces kernel launch overhead for leaf nodes.
    const int THRESHOLD = 32; // Define threshold for switching to sequential sort
    if (right - left + 1 <= THRESHOLD) { // Check if current chunk size is below threshold
        if (threadIdx.x == 0) { // Use only one thread for sequential sort
            // Simple selection sort for small chunk
             for (int i = left; i <= right; ++i) { // Outer loop for selection sort
                int min_idx = i; // Assume current index is minimum
                for (int j = i+1; j <= right; ++j) { // Inner loop to find true minimum
                    if (data[j] < data[min_idx]) min_idx = j; // Update minimum index if smaller element found
                }
                if (min_idx != i) gpu_swap(data[i], data[min_idx]); // Swap if new minimum found
            }
        }
        return; // Exit after sorting small chunk
    }

    // Partition Step
    // For large chunks, we use thread 0 (manager) to partition and launch children.
    // Note: Fully parallel partition (using all threads in block or prefix sum) 
    // requires significant boilerplate (Atomic Scan). 
    // For this assignment, increasing THRESHOLD is the primary optimization.
    
    if (threadIdx.x == 0) { // Manager thread handles partitioning (could be optimized further)
        int pivot = data[right]; // Select pivot element (last element)
        int i = left - 1; // Initialize smaller element index
        
        // Serial Partition on Manager Thread (Bottleneck for huge arrays, but fast enough with large Threshold)
        for (int j = left; j <= right - 1; j++) { // Iterate through the range
            if (data[j] < pivot) { // If current element is smaller than pivot
                i++; // Increment index of smaller element
                gpu_swap(data[i], data[j]); // Swap elements
            }
        }
        gpu_swap(data[i + 1], data[right]); // Place pivot in correct position
        int partition_idx = i + 1; // Store pivot index
        
        // Launch child kernels for sub-arrays
        // Use non-blocking streams to allow concurrent execution
        cudaStream_t s1, s2; // Declare two CUDA streams
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking); // Create non-blocking stream 1
        cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking); // Create non-blocking stream 2
        
        // Dynamic Parallelism: Launch new kernels
        quick_sort_partition_and_launch<<<1, 1, 0, s1>>>(data, left, partition_idx - 1, depth + 1); // Recursively sort left part
        quick_sort_partition_and_launch<<<1, 1, 0, s2>>>(data, partition_idx + 1, right, depth + 1); // Recursively sort right part
        
        cudaStreamDestroy(s1); // Destroy stream 1
        cudaStreamDestroy(s2); // Destroy stream 2
    }
}

// Wrapper to match old name if needed or update main
__global__ void cdp_quicksort_kernel(int* data, int left, int right, int depth) { // Wrapper kernel
    quick_sort_partition_and_launch<<<1, 1>>>(data, left, right, depth); // Helper launch
}


// =============================================================
// HEAP SORT (Parallel Build, Sequential Extract)
// =============================================================

__device__ void gpu_heapify(int* arr, int n, int i) { // Device function to heapify a subtree
    int largest = i; // Initialize largest as root
    int left = 2 * i + 1; // Calculate left child index
    int right = 2 * i + 2; // Calculate right child index

    if (left < n && arr[left] > arr[largest]) // Check left child
        largest = left; // Update largest

    if (right < n && arr[right] > arr[largest]) // Check right child
        largest = right; // Update largest

    if (largest != i) { // If largest is not root
        gpu_swap(arr[i], arr[largest]); // Swap
        gpu_heapify(arr, n, largest); // Recursively heapify
    }
}

// Kernel for one level of heap build
__global__ void build_heap_level_kernel(int* arr, int n, int start_idx, int end_idx) { // Kernel to heapify a range of nodes
    int idx = start_idx + blockIdx.x * blockDim.x + threadIdx.x; // Calculate global index
    if (idx >= start_idx && idx <= end_idx) { // Check bounds
        gpu_heapify(arr, n, idx); // Call heapify for this node
    }
}

// Host function to orchestrate parallel build
void invoke_gpu_build_heap(int* d_arr, int n) { // Host function to control heap construction
    // Process levels from bottom-up
    // Last non-leaf node is at n/2 - 1.
    // We can group independent nodes. Independent nodes are loose.
    // Actually, just calling heapify for i = n/2-1 down to 0 is the logic.
    // Can we parallelize? 
    // Yes, but dependencies exist: `heapify(i)` needs `2i+1` and `2i+2` to be heaps.
    // So we must convert strictly level by level (from bottom to top).
    // Level d (d=0 is root): Contains nodes 2^d to 2^(d+1)-1.
    // Max level ~ log2(N).
    // We iterate level from Max down to 0.
    
    int last_node = n / 2 - 1; // Calculate index of last non-leaf node
    if (last_node < 0) return; // If no non-leaf nodes, return
    
    // Just find the level of last_node?
    // Actually simpler: 
    // Just iterate `i` from `n/2 - 1` down to 0? No, that's serial.
    // We need to batch. 
    // Batch 1: All nodes at max_depth-1.
    // Batch 2: All nodes at max_depth-2. (These depend on Batch 1).
    // ...
    
    // Determining start/end indices for levels is tricky because array is a complete binary tree.
    // Simple heuristic: 
    // Process indices in batches? 
    // No, standard array heap property is strictly hierarchical. 
    // Node `k` depends on `2k+1, 2k+2`.
    // Valid Parallel Sets:
    // Any set of nodes such that no node is an ancestor of another in the set.
    // For a complete binary tree, all nodes at the same depth are independent.
    
    // So we iterate `depth` from `max_depth` down to 0.
    // For each `depth`, launch kernel for all nodes at this depth.
    
    int h = 0; // Initialize height
    while ((1 << (h + 1)) <= n) h++; // approx height calculation
    
    for (int d = h; d >= 0; d--) { // Loop from bottom level up to root
        int start_idx = (1 << d) - 1; // Calculate start index of level
        int end_idx = (1 << (d + 1)) - 2; // Calculate end index of level
        if (start_idx > last_node) continue; // Skip if level has no non-leaf nodes
        if (end_idx > last_node) end_idx = last_node; // Clamp end index to last non-leaf node
        
        int count = end_idx - start_idx + 1; // Calculate number of nodes to process at this level
        if (count > 0) { // If there are nodes to process
            int threads = 256; // Define threads per block
            int blocks = (count + threads - 1) / threads; // Calculate number of blocks
            build_heap_level_kernel<<<blocks, threads>>>(d_arr, n, start_idx, end_idx); // Launch kernel for this level
            cudaDeviceSynchronize(); // Wait for level to complete (essential for dependency)
        }
    }
}

// Kernel for sequential extraction (since it's hard to parallelize)
// Run on 1 thread? Very slow.
// User asks for comparison. If GPU is slower, that's a valid result.
__global__ void one_thread_heap_sort_extract(int* arr, int n) { // Kernel for extraction phase
    // Expects heap is already built
    for (int i = n - 1; i > 0; i--) { // Loop to extract max element repeatedly
        gpu_swap(arr[0], arr[i]); // Swap root (max) with last element
        gpu_heapify(arr, i, 0); // Restore heap property for the reduced heap
    }
}


#endif // SORTERS_GPU_CUH // End of include guard
