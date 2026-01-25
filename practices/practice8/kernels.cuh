#pragma once // Директива preprocessor untuk memastikan header file dincludekan hanya satu kali (хотя лучше использовать ifndef/define для переносимости)
#include <cuda_runtime.h> // Подключаем библиотеку Runtime API для CUDA
#include <cstdio> // Подключаем библиотеку для ввода-вывода (printf и т.д.)
#include <cstdlib> // Подключаем стандартную библиотеку (exit и т.д.)

// CUDA Ядро для умножения элементов на 2
// data: указатель на массив данных
// n: количество элементов в массиве
__global__ void multiply_kernel(float* data, int n);

// Вспомогательная функция для проверки ошибок CUDA
// result: код ошибки
// func: имя функции, вызвавшей ошибку
// file: имя файла, где произошла ошибка
// line: номер строки, где произошла ошибка
void checkCuda(cudaError_t result, char const* const func, const char* const file, int const line);

// Макрос для удобного вызова проверки ошибок
#define checkCudaErrors(val) checkCuda((val), #val, __FILE__, __LINE__)
