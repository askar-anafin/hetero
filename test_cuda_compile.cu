
#include <stdio.h>

__global__ void helloCuda() {
    printf("Hello from GPU!\n");
}

int main() {
    helloCuda<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("Hello from CPU!\n");
    return 0;
}
