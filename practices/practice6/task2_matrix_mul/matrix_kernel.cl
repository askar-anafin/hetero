// Define the kernel function 'matrix_mul' with arguments:
// A: Pointer to the first input matrix in global memory (read-only)
// B: Pointer to the second input matrix in global memory (read-only)
// C: Pointer to the output matrix in global memory (write-only)
// width: The width (and height) of the square matrices
__kernel void matrix_mul(__global const float* A, __global const float* B, __global float* C, int width) {
    // Get the global ID in the x-dimension, which corresponds to the column index
    int col = get_global_id(0);
    // Get the global ID in the y-dimension, which corresponds to the row index
    int row = get_global_id(1);

    // Check boundary conditions to ensure we don't access memory out of bounds
    if (row < width && col < width) {
        // Variable to store the computed sum for the element at C[row][col]
        float sum = 0.0f;
        // Iterate over the shared dimension 'k' to calculate the dot product of row 'row' of A and col 'col' of B
        for (int k = 0; k < width; k++) {
            // Multiply element from A and B and accumulate the result
            // A is accessed by row-major: A[row * width + k]
            // B is accessed by row-major: B[k * width + col]
            sum += A[row * width + k] * B[k * width + col];
        }
        // Store the final computed sum in the output matrix C at the corresponding position
        C[row * width + col] = sum;
    }
}
