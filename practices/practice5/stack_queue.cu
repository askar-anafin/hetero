#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// --- Stack Implementation ---
struct Stack {
    int *data;
    int *top; // Use a pointer to an int for atomic operations
    int capacity;

    __device__ void init(int *buffer, int *top_ptr, int size) {
        data = buffer;
        top = top_ptr;
        capacity = size;
    }

    __device__ bool push(int value) {
        int pos = atomicAdd(top, 1);
        if (pos < capacity) {
            data[pos] = value;
            return true;
        }
        // If we overflow, we should decrement back or handle it.
        // For this task, we assume capacity is enough.
        return false;
    }

    __device__ bool pop(int *value) {
        int pos = atomicSub(top, 1) - 1;
        if (pos >= 0) {
            *value = data[pos];
            return true;
        }
        // If we underflow, reset top to 0 if it went negative
        // However, atomicSub returns old value. 
        // If old top was 0, pos is -1. 
        return false;
    }
};

// --- Queue Implementation ---
struct Queue {
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
        if (pos < capacity) {
            data[pos] = value;
            return true;
        }
        return false;
    }

    __device__ bool dequeue(int *value) {
        int pos = atomicAdd(head, 1);
        // Note: This simple queue design doesn't handle wrap-around (circularity)
        // or the case where tail hasn't reached the data yet (pos >= tail).
        // For the task requirement, we follow the provided snippet.
        if (pos < *tail) {
            *value = data[pos];
            return true;
        }
        return false;
    }
};

// --- Kernels ---

__global__ void stackTestKernel(int *data, int *top, int capacity, int n_ops) {
    Stack s;
    s.init(data, top, capacity);
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_ops) {
        s.push(tid);
    }
}

__global__ void stackPopKernel(int *data, int *top, int capacity, int n_ops, int *results) {
    Stack s;
    s.init(data, top, capacity);
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_ops) {
        int val;
        if (s.pop(&val)) {
            results[tid] = val;
        } else {
            results[tid] = -1;
        }
    }
}

__global__ void queueTestKernel(int *data, int *head, int *tail, int capacity, int n_ops) {
    Queue q;
    q.init(data, head, tail, capacity);
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_ops) {
        q.enqueue(tid);
    }
}

__global__ void queueDequeueKernel(int *data, int *head, int *tail, int capacity, int n_ops, int *results) {
    Queue q;
    q.init(data, head, tail, capacity);
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_ops) {
        int val;
        if (q.dequeue(&val)) {
            results[tid] = val;
        } else {
            results[tid] = -1;
        }
    }
}

// --- Sequential implementations for comparison ---
void sequentialStack(int n_ops) {
    int *data = new int[n_ops];
    int top = -1;
    for (int i = 0; i < n_ops; ++i) {
        data[++top] = i;
    }
    for (int i = 0; i < n_ops; ++i) {
        int val = data[top--];
    }
    delete[] data;
}

void sequentialQueue(int n_ops) {
    int *data = new int[n_ops];
    int head = 0;
    int tail = 0;
    for (int i = 0; i < n_ops; ++i) {
        data[tail++] = i;
    }
    for (int i = 0; i < n_ops; ++i) {
        int val = data[head++];
    }
    delete[] data;
}

int main() {
    const int N = 1000000; // 1 million operations
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate Device Memory
    int *d_stackData, *d_stackTop;
    int *d_queueData, *d_queueHead, *d_queueTail;
    int *d_results;

    cudaMalloc(&d_stackData, N * sizeof(int));
    cudaMalloc(&d_stackTop, sizeof(int));
    cudaMalloc(&d_queueData, N * sizeof(int));
    cudaMalloc(&d_queueHead, sizeof(int));
    cudaMalloc(&d_queueTail, sizeof(int));
    cudaMalloc(&d_results, N * sizeof(int));

    // Initialize
    int h_zero = 0;
    int h_neg_one = -1;
    cudaMemcpy(d_stackTop, &h_zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_queueHead, &h_zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_queueTail, &h_zero, sizeof(int), cudaMemcpyHostToDevice);

    // --- Part 1: Stack ---
    std::cout << "Testing Parallel Stack..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    stackTestKernel<<<blocks, threadsPerBlock>>>(d_stackData, d_stackTop, N, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> stackPushTime = end - start;

    start = std::chrono::high_resolution_clock::now();
    stackPopKernel<<<blocks, threadsPerBlock>>>(d_stackData, d_stackTop, N, N, d_results);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> stackPopTime = end - start;

    int finalTop;
    cudaMemcpy(&finalTop, d_stackTop, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Stack Push Time: " << stackPushTime.count() << " ms" << std::endl;
    std::cout << "Stack Pop Time: " << stackPopTime.count() << " ms" << std::endl;
    std::cout << "Final Stack Top (expected 0): " << finalTop << std::endl;

    // --- Part 2: Queue ---
    std::cout << "\nTesting Parallel Queue..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    queueTestKernel<<<blocks, threadsPerBlock>>>(d_queueData, d_queueHead, d_queueTail, N, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> queueEnqueueTime = end - start;

    start = std::chrono::high_resolution_clock::now();
    queueDequeueKernel<<<blocks, threadsPerBlock>>>(d_queueData, d_queueHead, d_queueTail, N, N, d_results);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> queueDequeueTime = end - start;

    int finalHead, finalTail;
    cudaMemcpy(&finalHead, d_queueHead, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&finalTail, d_queueTail, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Queue Enqueue Time: " << queueEnqueueTime.count() << " ms" << std::endl;
    std::cout << "Queue Dequeue Time: " << queueDequeueTime.count() << " ms" << std::endl;
    std::cout << "Final Queue Head: " << finalHead << ", Tail: " << finalTail << std::endl;

    // --- Part 3: Sequential vs Parallel Comparison ---
    std::cout << "\nComparing with Sequential Implementations (N=" << N << ")..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    sequentialStack(N);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> seqStackTime = end - start;
    std::cout << "Sequential Stack Time: " << seqStackTime.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    sequentialQueue(N);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> seqQueueTime = end - start;
    std::cout << "Sequential Queue Time: " << seqQueueTime.count() << " ms" << std::endl;

    // Cleanup
    cudaFree(d_stackData);
    cudaFree(d_stackTop);
    cudaFree(d_queueData);
    cudaFree(d_queueHead);
    cudaFree(d_queueTail);
    cudaFree(d_results);

    return 0;
}
