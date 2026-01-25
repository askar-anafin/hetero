#include "kernels.cuh" // Подключаем заголовочный файл с объявлениями кернелов и утилит

// Реализация CUDA ядра для умножения элементов
// __global__ указывает, что функция выполняется на GPU и вызывается с CPU
__global__ void multiply_kernel(float* data, int n) {
    // Вычисляем глобальный индекс потока
    // blockIdx.x: индекс блока в сетке
    // blockDim.x: количество потоков в блоке
    // threadIdx.x: индекс потока внутри блока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Проверяем, что индекс находится в пределах размера массива
    if (idx < n) {
        data[idx] *= 2.0f; // Умножаем элемент массива на 2.0
    }
}

// Реализация функции проверки ошибок CUDA
void checkCuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    // Если результат не равен cudaSuccess (0)
    if (result) {
        // Выводим сообщение об ошибке в стандартный поток ошибок
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
            file, line, static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        // Сбрасываем состояние устройства
        cudaDeviceReset();
        // Завершаем программу с кодом ошибки 99
        exit(99);
    }
}
