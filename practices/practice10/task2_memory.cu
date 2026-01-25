#include <iostream> // Подключение библиотеки ввода-вывода
#include <vector>   // Подключение библиотеки для работы с векторами
#include <cuda_runtime.h> // Подключение заголовочного файла CUDA Runtime API
#include <device_launch_parameters.h> // Подключение параметров запуска ядра

// Макрос для проверки ошибок CUDA
#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    }

// 1. Коалесцированный (объединенный) доступ
// Потоки обращаются к последовательным адресам памяти, что позволяет загружать данные одной транзакцией
__global__ void k_coalesced(const float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Вычисление глобального индекса потока
    if (tid < n) { // Проверка выхода за границы массива
        // Коалесцированное чтение и запись (чтение input[tid] и запись в output[tid])
        output[tid] = input[tid] * 2.0f; // Умножение на 2
    }
}

// 2. Некоалесцированный (с шагом) доступ
// Потоки обращаются к памяти с шагом (stride), вызывая множество транзакций памяти
__global__ void k_non_coalesced(const float* input, float* output, int n, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Вычисление глобального индекса потока
    
    // Мы используем strided доступ для симуляции плохой работы с памятью
    // Каждый поток обращается не к соседнему элементу, а с большим смещением
    
    if (tid < n) { // Проверка выхода за границы
        // Вычисляем индекс с "прыжками" по памяти
        // (tid * stride) % n обеспечивает разброс обращений по всему массиву
        int idx = (tid * stride) % n; 
        output[idx] = input[idx] * 2.0f; // Чтение и запись по разбросанным индексам
    }
}

// 3. Оптимизация с использованием разделяемой памяти (Shared Memory)
// Загрузка блока данных в быструю разделяемую память, обработка и запись обратно
__global__ void k_shared(const float* input, float* output, int n) {
    extern __shared__ float s_data[]; // Объявление массива в разделяемой памяти (размер задается при запуске)
    
    int tid = threadIdx.x; // Локальный индекс потока внутри блока
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x; // Глобальный индекс потока
    
    if (global_idx < n) { // Проверка границ
        // Коалесцированная загрузка из глобальной памяти в разделяемую
        s_data[tid] = input[global_idx]; 
    }
    __syncthreads(); // Барьерная синхронизация: ждем, пока все потоки загрузят данные
    
    if (global_idx < n) { // Проверка границ
        // Обработка данных в разделяемой памяти (намного быстрее глобальной)
        float val = s_data[tid]; 
        val = val * 2.0f; // Выполнение операции
        
        // Коалесцированная запись результата обратно в глобальную память
        output[global_idx] = val; 
    }
}

// Функция для запуска и замера времени ядра
void benchmark(const char* name, void (*kernel)(const float*, float*, int), int grid, int block, int n, float* d_in, float* d_out) {
    cudaEvent_t start, stop; // Объявление событий CUDA для тайминга
    cudaEventCreate(&start); // Создание события старта
    cudaEventCreate(&stop);  // Создание события стопа

    cudaEventRecord(start); // Запись метки времени перед запуском
    // Запуск ядра с заданными параметрами. Размер shared memory = block * sizeof(float)
    kernel<<<grid, block, block * sizeof(float)>>>(d_in, d_out, n);
    cudaEventRecord(stop);  // Запись метки времени после запуска
    
    cudaEventSynchronize(stop); // Ожидание завершения работы GPU
    float milliseconds = 0; // Переменная для времени
    cudaEventElapsedTime(&milliseconds, start, stop); // Вычисление прошедшего времени
    
    std::cout << "Kernel: " << name << " | Time: " << milliseconds << " ms" << std::endl; // Вывод результата
    CHECK_CUDA(cudaGetLastError()); // Проверка на ошибки при запуске ядра
    
    cudaEventDestroy(start); // Уничтожение события
    cudaEventDestroy(stop);  // Уничтожение события
}

// Перегрузка функции бенчмарка для strided ядра (принимает параметр stride)
void benchmark_strided(const char* name, int stride, int grid, int block, int n, float* d_in, float* d_out) {
    cudaEvent_t start, stop; // Объявление событий
    cudaEventCreate(&start); // Создание
    cudaEventCreate(&stop);  // Создание

    cudaEventRecord(start); // Старт замера
    // Запуск некоалесцированного ядра с параметром stride
    k_non_coalesced<<<grid, block>>>(d_in, d_out, n, stride);
    cudaEventRecord(stop);  // Стоп замера
    
    cudaEventSynchronize(stop); // Синхронизация
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop); // Расчет времени
    
    std::cout << "Kernel: " << name << " (Stride " << stride << ") | Time: " << milliseconds << " ms" << std::endl; // Вывод
    CHECK_CUDA(cudaGetLastError()); // Проверка ошибок

    cudaEventDestroy(start); // Очистка ресурсов
    cudaEventDestroy(stop);  // Очистка ресурсов
}

int main() {
    int n = 1 << 24; // Объем данных: 2^24 = 16 миллионов элементов
    size_t size = n * sizeof(float); // Размер данных в байтах
    std::cout << "Processing " << n << " elements (" << (size / 1024 / 1024) << " MB)..." << std::endl; // Информация о запуске

    float *h_in, *h_out; // Указатели на хост-память
    CHECK_CUDA(cudaMallocHost(&h_in, size)); // Выделение pinned (закрепленной) памяти на хосте для ускорения передачи
    CHECK_CUDA(cudaMallocHost(&h_out, size)); // Выделение памяти под результат

    for (int i = 0; i < n; ++i) h_in[i] = 1.0f; // Инициализация входных данных

    float *d_in, *d_out; // Указатели на память устройства (GPU)
    CHECK_CUDA(cudaMalloc(&d_in, size)); // Выделение глобальной памяти на GPU
    CHECK_CUDA(cudaMalloc(&d_out, size)); // Выделение памяти под результат на GPU

    CHECK_CUDA(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice)); // Копирование данных с CPU на GPU

    int blockSize = 256; // Количество потоков в блоке
    int gridSize = (n + blockSize - 1) / blockSize; // Количество блоков в сетке

    std::cout << "------------------------------------------------" << std::endl; // Разделитель
    // Тест 1: Коалесцированный доступ (оптимальный)
    benchmark("Coalesced", k_coalesced, gridSize, blockSize, n, d_in, d_out);
    
    // Тест 2: Использование разделяемой памяти
    benchmark("Shared Memory", k_shared, gridSize, blockSize, n, d_in, d_out);

    // Тест 3: Некоалесцированный доступ (плохой паттерн)
    // Stride 1009 (простое число) создает хаотичный доступ, "убивающий" кэш и объединение запросов
    benchmark_strided("Non-Coalesced", 1009, gridSize, blockSize, n, d_in, d_out); 
    // Stride 32 (размер варпа), тоже может быть неэффективным, но часто попадает в одну линию кэша L2
    benchmark_strided("Non-Coalesced", 32, gridSize, blockSize, n, d_in, d_out);   
    std::cout << "------------------------------------------------" << std::endl; // Разделитель

    CHECK_CUDA(cudaFree(d_in)); // Освобождение памяти GPU
    CHECK_CUDA(cudaFree(d_out)); // Освобождение памяти GPU
    CHECK_CUDA(cudaFreeHost(h_in)); // Освобождение хост-памяти
    CHECK_CUDA(cudaFreeHost(h_out)); // Освобождение хост-памяти

    return 0; // Завершение программы
}
