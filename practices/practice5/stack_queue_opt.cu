#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>

// --- Optimized Stack with Shared Memory ---
// In a real scenario, shared memory is useful if a block performs many operations
// and then flushes to global memory. Here we'll simulate a block-local stack.
__global__ void sharedStackKernel(int *global_data, int *global_top, int n_ops_per_thread) {
    extern __shared__ int s_data[];
    __shared__ int s_top;

    if (threadIdx.x == 0) s_top = 0;
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread pushes some values to shared stack
    for (int i = 0; i < n_ops_per_thread; ++i) {
        int pos = atomicAdd(&s_top, 1);
        if (pos < 1024) { // Assume shared size
            s_data[pos] = tid * 100 + i;
        }
    }
    __syncthreads();

    // In a real app, we might do something with s_data and then exit.
    // To show "optimization", we just show we can use shared memory for atomics.
}

// --- Multi-Producer Multi-Consumer (MPMC) Queue ---
// A true MPMC queue often uses a circular buffer with head/tail atomics.
struct MPMCQueue {
    int *data;
    int *head;
    int *tail;
    int capacity;

    __device__ void init(int *buffer, int *head_ptr, int *tail_ptr, int size) {
        data = buffer;
        head = head_ptr;
        tail = tail_ptr;
        capacity = size;
    }

    __device__ bool enqueue(int value) {
        int pos = atomicAdd(tail, 1);
        // Standard check for overflow
        if (pos < capacity) {
            // In a circular queue, we'd use pos % capacity
            data[pos % capacity] = value;
            return true;
        }
        return false;
    }

    __device__ bool dequeue(int *value) {
        // Spin or check if tail > head
        // For simple MPMC, we just atomic increment head
        int pos = atomicAdd(head, 1);
        if (pos < *tail) {
            *value = data[pos % capacity];
            return true;
        }
        return false;
    }
};

__global__ void mpmcQueueKernel(int *data, int *head, int *tail, int capacity, int n_ops) {
    MPMCQueue q;
    q.init(data, head, tail, capacity);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_ops) {
        q.enqueue(tid);
        int val;
        q.dequeue(&val);
    }
}

int main() {
    const int N = 1000000;
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    int *d_data, *d_head, *d_tail;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_head, sizeof(int));
    cudaMalloc(&d_tail, sizeof(int));

    int h_zero = 0;
    cudaMemcpy(d_head, &h_zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tail, &h_zero, sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "Testing Optimized MPMC Queue (Enqueue + Dequeue in one kernel)..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    mpmcQueueKernel<<<blocks, threadsPerBlock>>>(d_data, d_head, d_tail, N, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> mpmcTime = end - start;
    std::cout << "MPMC Queue Time: " << mpmcTime.count() << " ms" << std::endl;

    std::cout << "Testing Shared Memory Stack (Simulation)..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    sharedStackKernel<<<blocks, threadsPerBlock, 1024 * sizeof(int)>>>(d_data, d_tail, 4);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> sharedStackTime = end - start;
    std::cout << "Shared Memory Stack Time: " << sharedStackTime.count() << " ms" << std::endl;

    cudaFree(d_data);
    cudaFree(d_head);
    cudaFree(d_tail);

    return 0;
}
