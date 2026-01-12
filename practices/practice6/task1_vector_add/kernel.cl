// Define the kernel function 'vector_add' with arguments:
// A: Pointer to the first input vector in global memory (read-only)
// B: Pointer to the second input vector in global memory (read-only)
// C: Pointer to the output vector in global memory (write-only)
__kernel void vector_add(__global const float* A, __global const float* B, __global float* C) {
    // Get the global ID of the current work-item in the 0th dimension (the index of the element to process)
    int id = get_global_id(0);
    // Perform element-wise addition: add element from A and B at index 'id' and store in C
    C[id] = A[id] + B[id];
}
