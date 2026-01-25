#include <iostream> // Подключение библиотеки ввода-вывода
#include <vector>   // Подключение вектора
#include <cmath>    // Математическая библиотека
#include <cuda_runtime.h> // API CUDA
#include <omp.h>    // API OpenMP

// Макрос проверки ошибок CUDA
#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    }

// Простое ядро: C = A + B с существенной нагрузкой
__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x; // Глобальный индекс
    if (i < n) {
        // Симуляция тяжелых вычислений
        float val = A[i] + B[i];
        for (int k = 0; k < 50; ++k) { // "Тяжелый" цикл для загрузки GPU
             val = sinf(val) * cosf(val) + val;
        }
        C[i] = val; // Запись результата
    }
}

// Функция для обработки на CPU (симуляция работы)
void cpu_process(std::vector<float>& data, int offset, int size) {
    // Используем OpenMP для распараллеливания работы CPU
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        float val = data[offset + i];
        // Имитация тяжелой математики
        for (int k = 0; k < 100; ++k) {
            val = sqrtf(val + 1.0f);
        }
        data[offset + i] = val;
    }
}

int main() {
    int n = 1 << 24; // 16 миллионов элементов
    size_t bytes = n * sizeof(float); // Размер в байтах
    std::cout << "Task 3: Hybrid CPU+GPU Processing (" << n << " elements)" << std::endl; // Вывод заголовка

    // Выделение Pinned Memory (закрепленной памяти) для асинхронных передач
    // Обычная (pageable) память не поддерживает полностью асинхронный memcpy
    float *h_a, *h_b, *h_c;
    CHECK_CUDA(cudaMallocHost((void**)&h_a, bytes)); // Вектор A
    CHECK_CUDA(cudaMallocHost((void**)&h_b, bytes)); // Вектор B
    CHECK_CUDA(cudaMallocHost((void**)&h_c, bytes)); // Вектор C (результат)

    // Инициализация данных
    for (int i = 0; i < n; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c; // Указатели на GPU
    CHECK_CUDA(cudaMalloc((void**)&d_a, bytes)); // Выделение A
    CHECK_CUDA(cudaMalloc((void**)&d_b, bytes)); // Выделение B
    CHECK_CUDA(cudaMalloc((void**)&d_c, bytes)); // Выделение C

    int n_streams = 4; // Количество потоков CUDA (Streams)
    cudaStream_t streams[4]; // Массив дескрипторов потоков
    for (int i = 0; i < n_streams; ++i) CHECK_CUDA(cudaStreamCreate(&streams[i])); // Создание потоков

    int streamSize = n / n_streams; // Размер чанка (куска) данных на один поток
    int streamBytes = streamSize * sizeof(float); // Размер чанка в байтах

    std::cout << "Streams: " << n_streams << " | Chunk Size: " << streamSize << std::endl; // Инфо

    // Бенчмарк гибридного исполнения
    cudaEvent_t start, stop; // События для тайминга
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Разогрев GPU (чтобы исключить накладные расходы инициализации из замера)
    vectorAdd<<<streamSize/256, 256, 0, streams[0]>>>(d_a, d_b, d_c, streamSize);
    CHECK_CUDA(cudaDeviceSynchronize()); // Ожидание завершения

    // --- Часть 1: Синхронное (Последовательное) исполнение ---
    // Сначала выполняем все на GPU, потом все на CPU. Без перекрытия.
    std::cout << "Starting Synchronous (Serial) Processing..." << std::endl;
    cudaEvent_t start_sync, stop_sync; // Создаем события для замера времени
    CHECK_CUDA(cudaEventCreate(&start_sync));
    CHECK_CUDA(cudaEventCreate(&stop_sync));

    CHECK_CUDA(cudaEventRecord(start_sync)); // Старт замера

    // 1. GPU Работа (Синхронно)
    // Копируем все данные A и B на девайс (блокирующая операция)
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    
    // Запускаем вычисления одним большим ядром (весь массив за раз)
    vectorAdd<<<n / 256, 256>>>(d_a, d_b, d_c, n);
    
    // Копируем результат обратно (блокирующая операция)
    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    
    // 2. CPU Работа (Синхронно, СТРОГО после завершения работы GPU)
    // Используем тот же объем вычислительной нагрузки, что и в гибридном тесте
    #pragma omp parallel for
    for (int k = 0; k < 1000000; ++k) {
        double x = std::sqrt(k * 1.0);
        if (x < 0) printf("?");
    }

    CHECK_CUDA(cudaEventRecord(stop_sync)); // Стоп замера
    CHECK_CUDA(cudaEventSynchronize(stop_sync)); // Синхронизация
    float time_sync = 0;
    CHECK_CUDA(cudaEventElapsedTime(&time_sync, start_sync, stop_sync)); // Расчет времени
    std::cout << "Mode: Synchronous | Time: " << time_sync << " ms" << std::endl;

    // --- Часть 2: Гибридное (Асинхронное) исполнение ---
    std::cout << "Starting Hybrid (Overlapped) Processing..." << std::endl;
    CHECK_CUDA(cudaEventRecord(start));

    // Пайплайн обработки: Запускаем работу на GPU по частям (чанками)
    // Благодаря Streams, передача данных (Memcpy) и вычисления (Kernel) могут перекрываться
    
    for (int i = 0; i < n_streams; ++i) { // Цикл по стримам
        int offset = i * streamSize; // Смещение для текущего чанка
        // Асинхронное копирование Host -> Device
        CHECK_CUDA(cudaMemcpyAsync(&d_a[offset], &h_a[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA(cudaMemcpyAsync(&d_b[offset], &h_b[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]));
        
        // Запуск ядра в соответствующем стриме
        vectorAdd<<<streamSize / 256, 256, 0, streams[i]>>>(&d_a[offset], &d_b[offset], &d_c[offset], streamSize);
        
        // Асинхронное копирование Device -> Host (результат)
        CHECK_CUDA(cudaMemcpyAsync(&h_c[offset], &d_c[offset], streamBytes, cudaMemcpyDeviceToHost, streams[i]));
    }
    
    // В этот момент GPU уже начал работать над первыми чанками.
    // Функция cudaMemcpyAsync возвращает управление немедленно (почти).
    // Поэтому CPU может начать свою работу ПАРАЛЛЕЛЬНО с GPU.
    
    std::cout << "GPU work enqueued. CPU starting parallel task..." << std::endl;
    
    // Независимая задача на CPU (та же самая нагрузка)
    #pragma omp parallel for
    for (int k = 0; k < 1000000; ++k) {
        double x = std::sqrt(k * 1.0); 
        if (x < 0) printf("?"); 
    }
    std::cout << "CPU task finished." << std::endl; 

    CHECK_CUDA(cudaDeviceSynchronize()); // Ждем, пока GPU закончит все стримы

    CHECK_CUDA(cudaEventRecord(stop)); // Засекаем конец общего времени
    CHECK_CUDA(cudaEventSynchronize(stop)); // Ждем записи события
    
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop)); // Считаем дельту
    std::cout << "Total Hybrid Time: " << milliseconds << " ms" << std::endl; // Вывод результата

    // Освобождение ресурсов
    for (int i = 0; i < n_streams; ++i) CHECK_CUDA(cudaStreamDestroy(streams[i])); // Удаление стримов
    CHECK_CUDA(cudaFree(d_a)); CHECK_CUDA(cudaFree(d_b)); CHECK_CUDA(cudaFree(d_c)); // Очистка GPU памяти
    CHECK_CUDA(cudaFreeHost(h_a)); CHECK_CUDA(cudaFreeHost(h_b)); CHECK_CUDA(cudaFreeHost(h_c)); // Очистка хост памяти

    return 0; // Конец
}
