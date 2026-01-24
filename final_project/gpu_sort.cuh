#ifndef GPU_SORT_CUH
#define GPU_SORT_CUH

#include <cuda_runtime.h> // Подключение основных функций CUDA Runtime
#include <device_launch_parameters.h> // Подключение параметров запуска ядра (threadIdx, blockIdx и т.д.)

// Ядро CUDA для одного шага битонической сортировки
__global__ void bitonic_sort_step(float *dev_values, int j, int k) {
    unsigned int i, ixj; /* Партнеры по сортировке: i и ixj */
    i = threadIdx.x + blockDim.x * blockIdx.x; // Вычисление глобального индекса нити
    ixj = i ^ j; // Вычисление индекса пары с использованием XOR

    /* Нити с меньшими id выполняют сортировку */
    if (ixj > i) { 
        if ((i & k) == 0) { // Определение направления сортировки (возрастание)
            /* Сортировка по возрастанию */
            if (dev_values[i] > dev_values[ixj]) { // Если элементы стоят в неправильном порядке
                float temp = dev_values[i]; // Временная переменная для обмена
                dev_values[i] = dev_values[ixj]; // Обмен значениями
                dev_values[ixj] = temp; // Обмен значениями
            }
        }
        if ((i & k) != 0) { // Определение направления сортировки (убывание)
            /* Сортировка по убыванию */
            if (dev_values[i] < dev_values[ixj]) { // Если элементы стоят в неправильном порядке
                float temp = dev_values[i]; // Временная переменная для обмена
                dev_values[i] = dev_values[ixj]; // Обмен значениями
                dev_values[ixj] = temp; // Обмен значениями
            }
        }
    }
}

// Хост-функция для управления битонической сортировкой на GPU
void gpuBitonicSort(float* dev_values, size_t size) {
    int threads = 512; // Количество нитей в блоке
    int blocks = (size + threads - 1) / threads; // Вычисление количества блоков

    // Внешний цикл по этапам битонической сортировки
    for (int k = 2; k <= size; k <<= 1) { 
        // Внутренний цикл по шагам внутри этапа
        for (int j = k >> 1; j > 0; j = j >> 1) { 
            // Запуск ядра для каждого шага
            bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
            cudaDeviceSynchronize(); // Ожидание завершения работы GPU перед следующим шагом
        }
    }
}

#endif // GPU_SORT_CUH
