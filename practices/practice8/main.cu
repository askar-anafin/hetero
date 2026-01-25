#include <iostream> // Подключаем библиотеку ввода-вывода (std::cout)
#include <vector> // Подключаем библиотеку векторов (std::vector)
#include <omp.h> // Подключаем библиотеку OpenMP для многопоточности на CPU
#include <chrono> // Подключаем библиотеку для работы со временем
#include <algorithm> // Подключаем библиотеку алгоритмов (если бы использовали, например, std::fill)
#include "kernels.cuh" // Подключаем заголовочный файл с CUDA-функциями

// Функция обработки на CPU с использованием OpenMP
void cpu_process(std::vector<float>& data) {
    int n = data.size(); // Получаем размер массива
    // Директива OpenMP для распараллеливания цикла for
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) { // Проходим по всем элементам
        data[i] *= 2.0f; // Умножаем текущий элемент на 2
    }
}

// Функция обработки на GPU
void gpu_process(const std::vector<float>& input, std::vector<float>& output) {
    int n = input.size(); // Получаем размер массива
    size_t bytes = n * sizeof(float); // Вычисляем размер данных в байтах
    float *d_data; // Указатель для памяти на устройстве (GPU)

    // Выделяем память на GPU и проверяем ошибки
    checkCudaErrors(cudaMalloc(&d_data, bytes));
    // Копируем данные из хоста (CPU) на устройство (GPU)
    checkCudaErrors(cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice));

    int blockSize = 256; // Задаем размер блока (количество потоков в блоке)
    int gridSize = (n + blockSize - 1) / blockSize; // Вычисляем размер сетки (количество блоков)

    // Запускаем ядро CUDA
    multiply_kernel<<<gridSize, blockSize>>>(d_data, n);
    // Проверяем ошибки запуска ядра (асинхронные ошибки)
    checkCudaErrors(cudaGetLastError());
    // Синхронизируем устройство (ждем завершения всех операций)
    checkCudaErrors(cudaDeviceSynchronize());

    // Копируем результат с устройства (GPU) обратно на хост (CPU)
    checkCudaErrors(cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost));
    // Освобождаем память на GPU
    checkCudaErrors(cudaFree(d_data));
}

// Гибридная обработка (CPU + GPU)
void hybrid_process(const std::vector<float>& input, std::vector<float>& output) {
    int n = input.size(); // Получаем размер массива
    int mid = n / 2; // Делим массив пополам (50/50)
    
    // Часть для CPU: индексы от 0 до mid
    // Часть для GPU: индексы от mid до n
    int n_gpu = n - mid; // Количество элементов для GPU
    size_t gpu_bytes = n_gpu * sizeof(float); // Размер данных для GPU в байтах
    float *d_data_part; // Указатель для памяти части GPU

    // Выделяем память на GPU для второй половины данных
    checkCudaErrors(cudaMalloc(&d_data_part, gpu_bytes));
    
    // Создаем поток CUDA для асинхронных операций (чтобы перекрыть вычисления)
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    // Асинхронно копируем вторую половину данных на GPU в созданном потоке
    checkCudaErrors(cudaMemcpyAsync(d_data_part, input.data() + mid, gpu_bytes, cudaMemcpyHostToDevice, stream));

    // Настраиваем параметры запуска ядра
    int blockSize = 256; // Размер блока
    int gridSize = (n_gpu + blockSize - 1) / blockSize; // Размер сетки
    // Запускаем ядро асинхронно в том же потоке
    multiply_kernel<<<gridSize, blockSize, 0, stream>>>(d_data_part, n_gpu);

    // Выполняем обработку на CPU, пока GPU работает
    // OpenMP распараллеливает цикл обработки первой половины массива
    #pragma omp parallel for
    for (int i = 0; i < mid; ++i) { // Проходим до mid
        output[i] = input[i] * 2.0f; // Записываем результат сразу в выходной массив
    }

    // Ожидаем завершения копирования (копируем результат GPU обратно)
    // Это тоже асинхронная операция в потоке, она начнется после завершения ядра
    checkCudaErrors(cudaMemcpyAsync(output.data() + mid, d_data_part, gpu_bytes, cudaMemcpyDeviceToHost, stream));
    
    // Синхронизируем поток (ждем завершения всех операций в потоке)
    checkCudaErrors(cudaStreamSynchronize(stream));

    // Уничтожаем поток
    checkCudaErrors(cudaStreamDestroy(stream));
    // Освобождаем память на GPU
    checkCudaErrors(cudaFree(d_data_part));
}

int main() {
    // "Прогреваем" контекст CUDA, чтобы избежать накладных расходов при первом вызове во время замеров
    cudaFree(0);

    int N = 1000000; // Размер массива (1 миллион элементов)
    std::cout << "Data size: " << N << " elements" << std::endl; // Выводим размер данных

    // Инициализируем входной вектор единицами
    std::vector<float> data(N, 1.0f);
    std::vector<float> cpu_result = data; // Создаем копию для результата CPU (инициализируем данными)
    std::vector<float> gpu_result(N); // Вектор для результата GPU
    std::vector<float> hybrid_result(N); // Вектор для результата гибридного режима

    // Задание 1: Замер времени CPU
    auto start = std::chrono::high_resolution_clock::now(); // Засекаем начало
    cpu_process(cpu_result); // Запускаем обработку на CPU
    auto end = std::chrono::high_resolution_clock::now(); // Засекаем конец
    std::chrono::duration<double, std::milli> cpu_time = end - start; // Вычисляем длительность в мс
    std::cout << "CPU Time (OpenMP): " << cpu_time.count() << " ms" << std::endl; // Выводим время

    // Задание 2: Замер времени GPU
    start = std::chrono::high_resolution_clock::now(); // Засекаем начало
    gpu_process(data, gpu_result); // Запускаем обработку на GPU
    end = std::chrono::high_resolution_clock::now(); // Засекаем конец
    std::chrono::duration<double, std::milli> gpu_time = end - start; // Вычисляем длительность
    std::cout << "GPU Time: " << gpu_time.count() << " ms" << std::endl; // Выводим время

    // Задание 3: Замер времени Гибридного режима
    start = std::chrono::high_resolution_clock::now(); // Засекаем начало
    hybrid_process(data, hybrid_result); // Запускаем гибридную обработку
    end = std::chrono::high_resolution_clock::now(); // Засекаем конец
    std::chrono::duration<double, std::milli> hybrid_time = end - start; // Вычисляем длительность
    std::cout << "Hybrid Time: " << hybrid_time.count() << " ms" << std::endl; // Выводим время

    // Валидация результатов (проверка корректности)
    bool correct = true; // Флаг корректности
    for(int i=0; i<N; ++i) { // Проходим по всем элементам
        // Проверяем каждое значение с допуском 1e-5
        if(abs(cpu_result[i] - 2.0f) > 1e-5) correct = false;
        if(abs(gpu_result[i] - 2.0f) > 1e-5) correct = false;
        if(abs(hybrid_result[i] - 2.0f) > 1e-5) correct = false;
    }

    // Выводим результат валидации
    if(correct) std::cout << "All results validated successfully!" << std::endl;
    else std::cout << "Validation FAILED!" << std::endl;

    // Задание 4: Анализ производительности
    std::cout << "\nPerformance Analysis:" << std::endl;
    // Считаем ускорение (Speedup)
    std::cout << "Speedup GPU vs CPU: " << cpu_time.count() / gpu_time.count() << "x" << std::endl;
    std::cout << "Speedup Hybrid vs CPU: " << cpu_time.count() / hybrid_time.count() << "x" << std::endl;
    
    // Сравниваем гибридный режим и GPU
    if (hybrid_time.count() < gpu_time.count()) {
        std::cout << "Hybrid is faster than GPU by " << gpu_time.count() - hybrid_time.count() << " ms" << std::endl;
    } else {
        std::cout << "GPU is faster than Hybrid by " << hybrid_time.count() - gpu_time.count() << " ms (Overhead or small N?)" << std::endl;
    }

    return 0; // Завершаем программу
}
