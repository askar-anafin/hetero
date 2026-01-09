#include <iostream> // Include Input/Output Stream library for standard I/O operations
#include <cuda_runtime.h> // Include CUDA runtime library for GPU operations
#include <chrono> // Include Chrono library for time measurement

// --- Stack Implementation ---
struct Stack { // Define a structure for the Stack data structure
    int *data; // Pointer to the integer array holding stack elements
    int *top; // Pointer to an integer representing the index of the top element
    int capacity; // Integer to store the maximum number of elements the stack can hold

    // Device function to initialize the stack
    __device__ void init(int *buffer, int *top_ptr, int size) {
        data = buffer; // Assign the allocated memory buffer to the stack data pointer
        top = top_ptr; // Assign the allocated memory for top pointer
        capacity = size; // Set the capacity of the stack
    }

    // Device function to push a value onto the stack
    __device__ bool push(int value) {
        int pos = atomicAdd(top, 1); // Atomically increment the top index and get the old value (current position)
        if (pos < capacity) { // Check if the current position is within the stack's capacity
            data[pos] = value; // Store the value at the current position in the data array
            return true; // Return true indicating successful push
        }
        // If we overflow, we should decrement back or handle it.
        // For this task, we assume capacity is enough.
        return false; // Return false indicating stack overflow (push failed)
    }

    // Device function to pop a value from the stack
    __device__ bool pop(int *value) {
        int pos = atomicSub(top, 1) - 1; // Atomically decrement the top index and calculate the position to pop from
        if (pos >= 0) { // Check if the position is valid (non-negative)
            *value = data[pos]; // Retrieve the value from the data array at the calculated position
            return true; // Return true indicating successful pop
        }
        // If we underflow, reset top to 0 if it went negative
        // However, atomicSub returns old value. 
        // If old top was 0, pos is -1. 
        return false; // Return false indicating stack underflow (pop failed)
    }
};

// --- Queue Implementation ---
struct Queue { // Define a structure for the Queue data structure
    int *data; // Pointer to the integer array holding queue elements
    int *head; // Pointer to an integer representing the head (front) of the queue
    int *tail; // Pointer to an integer representing the tail (back) of the queue
    int capacity; // Integer to store the maximum number of elements the queue can hold

    // Device function to initialize the queue
    __device__ void init(int *buffer, int *head_ptr, int *tail_ptr, int size) {
        data = buffer; // Assign the allocated memory buffer to the queue data pointer
        head = head_ptr; // Assign the allocated memory for head pointer
        tail = tail_ptr; // Assign the allocated memory for tail pointer
        capacity = size; // Set the capacity of the queue
    }

    // Device function to enqueue (add) a value to the queue
    __device__ bool enqueue(int value) {
        int pos = atomicAdd(tail, 1); // Atomically increment the tail index to reserve a position
        if (pos < capacity) { // Check if the reserved position is within the queue's capacity
            data[pos] = value; // Store the value at the reserved position in the data array
            return true; // Return true indicating successful enqueue
        }
        return false; // Return false indicating queue full/overflow
    }

    // Device function to dequeue (remove) a value from the queue
    __device__ bool dequeue(int *value) {
        int pos = atomicAdd(head, 1); // Atomically increment the head index to pick an element to remove
        // Note: This simple queue design doesn't handle wrap-around (circularity)
        // or the case where tail hasn't reached the data yet (pos >= tail).
        // For the task requirement, we follow the provided snippet.
        if (pos < *tail) { // Check if the head position is less than the current tail position (valid data exists)
            *value = data[pos]; // Retrieve the value from the data array at the head position
            return true; // Return true indicating successful dequeue
        }
        return false; // Return false indicating queue empty/underflow
    }
};

// --- Kernels ---

// Global kernel function to test the stack push operation
__global__ void stackTestKernel(int *data, int *top, int capacity, int n_ops) {
    Stack s; // Declare a Stack object
    s.init(data, top, capacity); // Initialize the stack with global memory pointers
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the global thread ID
    if (tid < n_ops) { // Check if the thread ID is within the number of operations
        s.push(tid); // Push the thread ID value onto the stack
    }
}

// Global kernel function to test the stack pop operation
__global__ void stackPopKernel(int *data, int *top, int capacity, int n_ops, int *results) {
    Stack s; // Declare a Stack object
    s.init(data, top, capacity); // Initialize the stack with global memory pointers
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the global thread ID
    if (tid < n_ops) { // Check if the thread ID is within the number of operations
        int val; // Variable to store the popped value
        if (s.pop(&val)) { // Attempt to pop a value from the stack
            results[tid] = val; // Store the popped value in the results array
        } else {
            results[tid] = -1; // Store -1 if pop failed (underflow)
        }
    }
}

// Global kernel function to test the queue enqueue operation
__global__ void queueTestKernel(int *data, int *head, int *tail, int capacity, int n_ops) {
    Queue q; // Declare a Queue object
    q.init(data, head, tail, capacity); // Initialize the queue with global memory pointers
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the global thread ID
    if (tid < n_ops) { // Check if the thread ID is within the number of operations
        q.enqueue(tid); // Enqueue the thread ID value into the queue
    }
}

// Global kernel function to test the queue dequeue operation
__global__ void queueDequeueKernel(int *data, int *head, int *tail, int capacity, int n_ops, int *results) {
    Queue q; // Declare a Queue object
    q.init(data, head, tail, capacity); // Initialize the queue with global memory pointers
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the global thread ID
    if (tid < n_ops) { // Check if the thread ID is within the number of operations
        int val; // Variable to store the dequeued value
        if (q.dequeue(&val)) { // Attempt to dequeue a value from the queue
            results[tid] = val; // Store the dequeued value in the results array
        } else {
            results[tid] = -1; // Store -1 if dequeue failed (empty)
        }
    }
}

// --- Sequential implementations for comparison ---
void sequentialStack(int n_ops) { // Function for sequential stack operations on CPU
    int *data = new int[n_ops]; // Allocate memory for stack data on CPU
    int top = -1; // Initialize top index to -1 (empty stack)
    for (int i = 0; i < n_ops; ++i) { // Loop for n_ops iterations
        data[++top] = i; // Push 'i' onto the stack (increment top, then store)
    }
    for (int i = 0; i < n_ops; ++i) { // Loop for n_ops iterations
        int val = data[top--]; // Pop from the stack (retrieve, then decrement top)
    }
    delete[] data; // Free the allocated memory
}

void sequentialQueue(int n_ops) { // Function for sequential queue operations on CPU
    int *data = new int[n_ops]; // Allocate memory for queue data on CPU
    int head = 0; // Initialize head index to 0
    int tail = 0; // Initialize tail index to 0
    for (int i = 0; i < n_ops; ++i) { // Loop for n_ops iterations
        data[tail++] = i; // Enqueue 'i' (store at tail, then increment tail)
    }
    for (int i = 0; i < n_ops; ++i) { // Loop for n_ops iterations
        int val = data[head++]; // Dequeue (retrieve from head, then increment head)
    }
    delete[] data; // Free the allocated memory
}

int main() { // Main function, entry point of the program
    const int N = 1000000; // Define constant N for 1 million operations
    const int threadsPerBlock = 256; // Define the number of threads per block
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock; // Calculate the number of blocks needed

    // Allocate Device Memory
    int *d_stackData, *d_stackTop; // Pointers for stack data and top index on device
    int *d_queueData, *d_queueHead, *d_queueTail; // Pointers for queue data, head, and tail indices on device
    int *d_results; // Pointer for results array on device

    cudaMalloc(&d_stackData, N * sizeof(int)); // Allocate memory for stack data on GPU
    cudaMalloc(&d_stackTop, sizeof(int)); // Allocate memory for stack top pointer on GPU
    cudaMalloc(&d_queueData, N * sizeof(int)); // Allocate memory for queue data on GPU
    cudaMalloc(&d_queueHead, sizeof(int)); // Allocate memory for queue head pointer on GPU
    cudaMalloc(&d_queueTail, sizeof(int)); // Allocate memory for queue tail pointer on GPU
    cudaMalloc(&d_results, N * sizeof(int)); // Allocate memory for results on GPU

    // Initialize
    int h_zero = 0; // Host variable for initialization (0)
    int h_neg_one = -1; // Host variable for initialization (-1)
    cudaMemcpy(d_stackTop, &h_zero, sizeof(int), cudaMemcpyHostToDevice); // Initialize stack top to 0 on GPU
    cudaMemcpy(d_queueHead, &h_zero, sizeof(int), cudaMemcpyHostToDevice); // Initialize queue head to 0 on GPU
    cudaMemcpy(d_queueTail, &h_zero, sizeof(int), cudaMemcpyHostToDevice); // Initialize queue tail to 0 on GPU

    // --- Part 1: Stack ---
    std::cout << "Testing Parallel Stack..." << std::endl; // Print message indicating stack testing
    auto start = std::chrono::high_resolution_clock::now(); // Start timer for stack push
    stackTestKernel<<<blocks, threadsPerBlock>>>(d_stackData, d_stackTop, N, N); // Launch stack push kernel
    cudaDeviceSynchronize(); // Wait for kernel to complete
    auto end = std::chrono::high_resolution_clock::now(); // Stop timer for stack push
    std::chrono::duration<double, std::milli> stackPushTime = end - start; // Calculate duration for stack push

    start = std::chrono::high_resolution_clock::now(); // Start timer for stack pop
    stackPopKernel<<<blocks, threadsPerBlock>>>(d_stackData, d_stackTop, N, N, d_results); // Launch stack pop kernel
    cudaDeviceSynchronize(); // Wait for kernel to complete
    end = std::chrono::high_resolution_clock::now(); // Stop timer for stack pop
    std::chrono::duration<double, std::milli> stackPopTime = end - start; // Calculate duration for stack pop

    int finalTop; // Variable to store final top value
    cudaMemcpy(&finalTop, d_stackTop, sizeof(int), cudaMemcpyDeviceToHost); // Copy final top value from device to host
    std::cout << "Stack Push Time: " << stackPushTime.count() << " ms" << std::endl; // Print stack push time
    std::cout << "Stack Pop Time: " << stackPopTime.count() << " ms" << std::endl; // Print stack pop time
    std::cout << "Final Stack Top (expected 0): " << finalTop << std::endl; // Print final stack top value

    // --- Part 2: Queue ---
    std::cout << "\nTesting Parallel Queue..." << std::endl; // Print message indicating queue testing
    start = std::chrono::high_resolution_clock::now(); // Start timer for queue enqueue
    queueTestKernel<<<blocks, threadsPerBlock>>>(d_queueData, d_queueHead, d_queueTail, N, N); // Launch queue enqueue kernel
    cudaDeviceSynchronize(); // Wait for kernel to complete
    end = std::chrono::high_resolution_clock::now(); // Stop timer for queue enqueue
    std::chrono::duration<double, std::milli> queueEnqueueTime = end - start; // Calculate duration for queue enqueue

    start = std::chrono::high_resolution_clock::now(); // Start timer for queue dequeue
    queueDequeueKernel<<<blocks, threadsPerBlock>>>(d_queueData, d_queueHead, d_queueTail, N, N, d_results); // Launch queue dequeue kernel
    cudaDeviceSynchronize(); // Wait for kernel to complete
    end = std::chrono::high_resolution_clock::now(); // Stop timer for queue dequeue
    std::chrono::duration<double, std::milli> queueDequeueTime = end - start; // Calculate duration for queue dequeue

    int finalHead, finalTail; // Variables to store final head and tail values
    cudaMemcpy(&finalHead, d_queueHead, sizeof(int), cudaMemcpyDeviceToHost); // Copy final head from device to host
    cudaMemcpy(&finalTail, d_queueTail, sizeof(int), cudaMemcpyDeviceToHost); // Copy final tail from device to host
    std::cout << "Queue Enqueue Time: " << queueEnqueueTime.count() << " ms" << std::endl; // Print queue enqueue time
    std::cout << "Queue Dequeue Time: " << queueDequeueTime.count() << " ms" << std::endl; // Print queue dequeue time
    std::cout << "Final Queue Head: " << finalHead << ", Tail: " << finalTail << std::endl; // Print final queue head and tail

    // --- Part 3: Sequential vs Parallel Comparison ---
    std::cout << "\nComparing with Sequential Implementations (N=" << N << ")..." << std::endl; // Print comparison message
    start = std::chrono::high_resolution_clock::now(); // Start timer for sequential stack
    sequentialStack(N); // Run sequential stack operations
    end = std::chrono::high_resolution_clock::now(); // Stop timer for sequential stack
    std::chrono::duration<double, std::milli> seqStackTime = end - start; // Calculate sequential stack duration
    std::cout << "Sequential Stack Time: " << seqStackTime.count() << " ms" << std::endl; // Print sequential stack time

    start = std::chrono::high_resolution_clock::now(); // Start timer for sequential queue
    sequentialQueue(N); // Run sequential queue operations
    end = std::chrono::high_resolution_clock::now(); // Stop timer for sequential queue
    std::chrono::duration<double, std::milli> seqQueueTime = end - start; // Calculate sequential queue duration
    std::cout << "Sequential Queue Time: " << seqQueueTime.count() << " ms" << std::endl; // Print sequential queue time

    // Cleanup
    cudaFree(d_stackData); // Free device memory for stack data
    cudaFree(d_stackTop); // Free device memory for stack top
    cudaFree(d_queueData); // Free device memory for queue data
    cudaFree(d_queueHead); // Free device memory for queue head
    cudaFree(d_queueTail); // Free device memory for queue tail
    cudaFree(d_results); // Free device memory for results

    return 0; // Return 0 to indicate successful execution
}
