#include <iostream> // Include the IO stream library for outputting text to console
#include <cuda_runtime.h> // Include CUDA runtime library for GPU memory and kernel management
#include <chrono> // Include Chrono library for high-resolution timing
#include <vector> // Include Vector library (though not strictly used in this device code, good for host)

// --- Optimized Stack with Shared Memory ---
// In a real scenario, shared memory is useful if a block performs many operations
// and then flushes to global memory. Here we'll simulate a block-local stack.
__global__ void sharedStackKernel(int *global_data, int *global_top, int n_ops_per_thread) {
    extern __shared__ int s_data[]; // Declare dynamic shared memory for stack data
    __shared__ int s_top; // Declare shared memory variable for stack pointer

    if (threadIdx.x == 0) s_top = 0; // Initialize shared stack top to 0 using the first thread
    __syncthreads(); // Synchronize all threads in the block to ensure s_top is initialized

    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global thread index
    
    // Each thread pushes some values to shared stack
    for (int i = 0; i < n_ops_per_thread; ++i) { // Loop through operations per thread
        int pos = atomicAdd(&s_top, 1); // Atomically increment shared stack top and capture position
        if (pos < 1024) { // Assume shared size is 1024 integers
            s_data[pos] = tid * 100 + i; // Store a unique value based on thread ID and iteration
        }
    }
    __syncthreads(); // Synchronize threads to ensure all writes to shared memory are done

    // In a real app, we might do something with s_data and then exit.
    // To show "optimization", we just show we can use shared memory for atomics.
}

// --- Multi-Producer Multi-Consumer (MPMC) Queue ---
// A true MPMC queue often uses a circular buffer with head/tail atomics.
struct MPMCQueue { // Define MPMCQueue structure
    int *data; // Pointer to queue data in global memory
    int *head; // Pointer to head index
    int *tail; // Pointer to tail index
    int capacity; // Maximum capacity of the queue

    // Helper to initialize properties
    __device__ void init(int *buffer, int *head_ptr, int *tail_ptr, int size) {
        data = buffer; // Set data buffer
        head = head_ptr; // Set head pointer
        tail = tail_ptr; // Set tail pointer
        capacity = size; // Set capacity
    }

    // Enqueue operation
    __device__ bool enqueue(int value) {
        int pos = atomicAdd(tail, 1); // Atomically reserve a slot at tail
        // Standard check for overflow
        if (pos < capacity) { // If within capacity... (simplified check)
            // In a circular queue, we'd use pos % capacity
            data[pos % capacity] = value; // Store value at modulo index (circular buffer)
            return true; // Enqueue successful
        }
        return false; // Queue full
    }

    // Dequeue operation
    __device__ bool dequeue(int *value) {
        // Spin or check if tail > head
        // For simple MPMC, we just atomic increment head
        int pos = atomicAdd(head, 1); // Atomically increment head to reserve item
        if (pos < *tail) { // Check if head is before tail (meaning valid data exists)
            *value = data[pos % capacity]; // Read value from modulo index
            return true; // Dequeue successful
        }
        return false; // Queue empty or contention issue
    }
};

// Global kernel to Test MPMC Queue
__global__ void mpmcQueueKernel(int *data, int *head, int *tail, int capacity, int n_ops) {
    MPMCQueue q; // Instantiate queue
    q.init(data, head, tail, capacity); // Initialize queue
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Get global thread ID
    if (tid < n_ops) { // Bounds check
        q.enqueue(tid); // Producer: Enqueue thread ID
        int val; // Variable to hold popped value
        q.dequeue(&val); // Consumer: immediately try to dequeue (simulation)
    }
}

int main() {
    const int N = 1000000; // Number of operations
    const int threadsPerBlock = 256; // Standard block size
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock; // Calculate grid size

    int *d_data, *d_head, *d_tail; // Device pointers
    cudaMalloc(&d_data, N * sizeof(int)); // Allocate data buffer
    cudaMalloc(&d_head, sizeof(int)); // Allocate head counter
    cudaMalloc(&d_tail, sizeof(int)); // Allocate tail counter

    int h_zero = 0; // Host zero for initialization
    cudaMemcpy(d_head, &h_zero, sizeof(int), cudaMemcpyHostToDevice); // Init head to 0
    cudaMemcpy(d_tail, &h_zero, sizeof(int), cudaMemcpyHostToDevice); // Init tail to 0

    std::cout << "Testing Optimized MPMC Queue (Enqueue + Dequeue in one kernel)..." << std::endl; // Status msg
    auto start = std::chrono::high_resolution_clock::now(); // Start timer
    mpmcQueueKernel<<<blocks, threadsPerBlock>>>(d_data, d_head, d_tail, N, N); // Launch Kernel
    cudaDeviceSynchronize(); // Wait for completion
    auto end = std::chrono::high_resolution_clock::now(); // Stop timer
    std::chrono::duration<double, std::milli> mpmcTime = end - start; // Calc duration
    std::cout << "MPMC Queue Time: " << mpmcTime.count() << " ms" << std::endl; // Print result

    std::cout << "Testing Shared Memory Stack (Simulation)..." << std::endl; // Status msg
    start = std::chrono::high_resolution_clock::now(); // Start timer
    sharedStackKernel<<<blocks, threadsPerBlock, 1024 * sizeof(int)>>>(d_data, d_tail, 4); // Launch shared mem kernel
    cudaDeviceSynchronize(); // Wait
    end = std::chrono::high_resolution_clock::now(); // Stop
    std::chrono::duration<double, std::milli> sharedStackTime = end - start; // Calc duration
    std::cout << "Shared Memory Stack Time: " << sharedStackTime.count() << " ms" << std::endl; // Print result

    cudaFree(d_data); // Free device memory
    cudaFree(d_head); // Free device memory
    cudaFree(d_tail); // Free device memory

    return 0; // Exit success
}
