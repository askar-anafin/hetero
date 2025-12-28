
/*
====================================================================================================
ЗАДАЧА 1. ВВЕДЕНИЕ В ГЕТЕРОГЕННУЮ ПАРАЛЛЕЛИЗАЦИЮ (Теоретическое задание)
====================================================================================================

1. Что такое гетерогенная параллелизация?
   Гетерогенная параллелизация — это стратегия вычислений, использующая более одного типа процессоров
   или ядер (например, CPU и GPU) для выполнения одной задачи. Система распределяет нагрузку между
   центральным процессором (CPU), который хорошо справляется со сложной логикой и последовательными
   задачами, и графическим процессором (GPU), который эффективен для массово-параллельных вычислений
   над большими объемами данных.

2. Различия между параллельными вычислениями на CPU и GPU:
   - Архитектура:
     * CPU: Мало мощных ядер (Latency-oriented). Оптимизирован для последовательной обработки, сложного
       управления потоком (ветвления) и кэширования.
     * GPU: Тысячи слабых ядер (Throughput-oriented). Оптимизирован для массового параллелизма,
       где одна и та же инструкция выполняется над множеством данных (SIMT/SIMD).
   - Контекст использования:
     * CPU лучше подходит для операционных систем, логики приложений, ввода-вывода.
     * GPU превосходит CPU в задачах обработки изображений, матричных вычислениях, deep learning.

3. Преимущества гетерогенной параллелизации:
   - Эффективность: Использование специализированных устройств для подходящих задач повышает общую
     производительность (закон Амдала).
   - Энергоэффективность: GPU часто обеспечивают больше FLOPS на ватт для параллельных задач.
   - Масштабируемость: Возможность подключать дополнительные ускорители для роста мощности.

4. Примеры реальных приложений:
   - Обучение нейросетей (TensorFlow, PyTorch): CPU готовит данные, GPU выполняет обучение.
   - Рендеринг компьютерной графики и игры.
   - Научное моделирование (погода, молекулярная динамика).
   - Криптография и майнинг.

====================================================================================================
*/

#include <iostream>     // Подключение библиотеки ввода-вывода / Include I/O library
#include <vector>       // Подключение библиотеки для использования векторов / Include vector container
#include <algorithm>    // Подключение библиотеки алгоритмов (min, max, sort) / Include algorithms
#include <ctime>        // Подключение библиотеки времени для srand / Include time library for random seed
#include <climits>      // Подключение констант INT_MAX, INT_MIN / Include limits constants
#include <iomanip>      // Подключение манипуляторов вывода / Include I/O manipulators
#include <chrono>       // Подключение библиотеки для точного замера времени / Include crono for timing

#ifdef _OPENMP          // Проверка, включена ли поддержка OpenMP / Check if OpenMP is enabled
#include <omp.h>        // Подключение библиотеки OpenMP / Include OpenMP library
#endif

// Helper to generate random array / Вспомогательная функция для генерации случайного массива
std::vector<int> generateRandomArray(int size) {
    std::vector<int> arr(size);        // Создание вектора заданного размера / Create vector of given size
    for (int i = 0; i < size; ++i) {   // Цикл по всем элементам / Loop through elements
        arr[i] = rand() % 100000;      // Заполнение случайными числами 0-99999 / Fill with random numbers
    }
    return arr;                        // Возврат заполненного массива / Return array
}

// Helper to measure execution time / Вспомогательная функция для замера времени выполнения
template<typename Func>
double measureTime(Func func) {
    auto start = std::chrono::high_resolution_clock::now(); // Засекаем время начала / Start timer
    func();                                                 // Выполняем переданную функцию / Execute function
    auto end = std::chrono::high_resolution_clock::now();   // Засекаем время конца / End timer
    std::chrono::duration<double, std::milli> duration = end - start; // Вычисляем длительность в мс / Calc duration in ms
    return duration.count();                                // Возвращаем количество миллисекунд / Return ms count
}

// =================================================================================================
// ЗАДАЧА 2. РАБОТА С МАССИВАМИ И OPENMP
// =================================================================================================

// Последовательная версия поиска минимума и максимума / Sequential Min/Max finder
void task2_sequential(const std::vector<int>& arr, int& minVal, int& maxVal) {
    minVal = INT_MAX;          // Инициализация минимума максимальным значением / Init min with max int
    maxVal = INT_MIN;          // Инициализация максимума минимальным значением / Init max with min int
    for (int val : arr) {      // Проход по всем элементам массива / Loop through array
        if (val < minVal) minVal = val; // Если текущий меньше min, обновляем / Update min if smaller
        if (val > maxVal) maxVal = val; // Если текущий больше max, обновляем / Update max if larger
    }
}

// Параллельная версия поиска мин/макс (совместимая с MSVC) / Parallel Min/Max (MSVC compatible)
void task2_parallel(const std::vector<int>& arr, int& minVal, int& maxVal) {
    int global_min = INT_MAX;   // Глобальный минимум / Global min for all threads
    int global_max = INT_MIN;   // Глобальный максимум / Global max for all threads
    int n = (int)arr.size();    // Размер массива / Array size

    #pragma omp parallel        // Начало параллельного региона / Start parallel region
    {
        int local_min = INT_MAX; // Локальный минимум потока / Thread-local min
        int local_max = INT_MIN; // Локальный максимум потока / Thread-local max

        #pragma omp for nowait  // Распределение итераций цикла без барьера / Distribute loop, no wait
        for (int i = 0; i < n; ++i) { // Цикл по массиву / Loop elements
            if (arr[i] < local_min) local_min = arr[i]; // Обновление локального минимума / Update local min
            if (arr[i] > local_max) local_max = arr[i]; // Обновление локального максимума / Update local max
        }

        #pragma omp critical    // Критическая секция (один поток за раз) / Critical section (atomic access)
        {
            if (local_min < global_min) global_min = local_min; // Обновление глобального минимума / Update global min
            if (local_max > global_max) global_max = local_max; // Обновление глобального максимума / Update global max
        }
    }
    minVal = global_min;        // Запись результата в выходную переменную / Write result to output
    maxVal = global_max;        // Запись результата в выходную переменную / Write result to output
}

void runTask2() {
    std::cout << "\n--- Task 2: OpenMP Min/Max (Array size: 10,000) ---\n"; // Вывод заголовка / Print header
    const int SIZE = 10000;     // Размер массива для теста / Test array size
    std::vector<int> arr = generateRandomArray(SIZE); // Генерация данных / Generate data
    
    int minSeq, maxSeq;         // Переменные для результатов (посл.) / Result storage (seq)
    double timeSeq = measureTime([&]() { task2_sequential(arr, minSeq, maxSeq); }); // Замер времени (посл.) / Timing (seq)
    
    int minPar, maxPar;         // Переменные для результатов (парал.) / Result storage (par)
    double timePar = measureTime([&]() { task2_parallel(arr, minPar, maxPar); });   // Замер времени (парал.) / Timing (par)

    std::cout << "Sequential: Min=" << minSeq << ", Max=" << maxSeq << " | Time: " << timeSeq << " ms\n"; // Результаты / Print seq results
    std::cout << "Parallel:   Min=" << minPar << ", Max=" << maxPar << " | Time: " << timePar << " ms\n"; // Результаты / Print par results
    
    if (minSeq == minPar && maxSeq == maxPar) { // Сравнение результатов / Compare results
        std::cout << "Results match. Speedup: " << timeSeq / timePar << "x\n"; // Совпадение и ускорение / Match info
    } else {
        std::cout << "ERROR: Results do not match!\n"; // Ошибка, если не совпали / Mismatch error
    }
}

// =================================================================================================
// ЗАДАЧА 3. ПАРАЛЛЕЛЬНАЯ СОРТИРОВКА С OPENMP (Selection Sort)
// =================================================================================================

// Последовательная сортировка выбором / Sequential Selection Sort
void selectionSortSequential(std::vector<int> arr) {
    int n = arr.size();                   // Получение размера массива / Get size
    for (int i = 0; i < n - 1; i++) {     // Внешний цикл от первого до предпоследнего / Outer loop
        int min_idx = i;                  // Индекс минимального элемента / Min value index
        for (int j = i + 1; j < n; j++) { // Внутренний цикл: поиск минимума / Inner loop: find min
            if (arr[j] < arr[min_idx])    // Сравнение текущего с минимальным / Compare current with min
                min_idx = j;              // Обновление индекса минимума / Update min index
        }
        std::swap(arr[i], arr[min_idx]);  // Обмен текущего элемента с найденным минимальным / Swap elements
    }
}

// Улучшенная параллельная сортировка выбором / Improved Parallel Selection Sort
void selectionSortParallelImproved(std::vector<int>& arr) {
    int n = arr.size();                   // Получение размера массива / Get size
    for (int i = 0; i < n - 1; i++) {     // Внешний цикл (последовательный!) / Outer loop (sequential!)
        struct Compare { int val; int index; }; // Структура для редукции / Struct for reduction
        Compare min_global = { arr[i], i };     // Глобальный минимум итерации / Global min for iteration

        #pragma omp parallel              // Параллельная область / Parallel region
        {
            Compare min_local = min_global; // Локальная копия минимума / Local min copy
            #pragma omp for nowait        // Распределение работы / Distribute work
            for (int j = i + 1; j < n; ++j) { // Цикл поиска минимума в остатке / Loop remaining part
                if (arr[j] < min_local.val) { // Сравнение / Compare
                    min_local.val = arr[j];   // Обновление значения / Update value
                    min_local.index = j;      // Обновление индекса / Update index
                }
            }

            #pragma omp critical          // Критическая секция для обновления глобального / Critical update
            {
                if (min_local.val < min_global.val) { // Проверка, нашли ли меньше / Check if smaller
                    min_global = min_local;           // Обновление глобального / Update global
                }
            }
        }
        if (min_global.index != i) {      // Если найден новый минимум / If new min found
            std::swap(arr[i], arr[min_global.index]); // Обмен элементов / Swap elements
        }
    }
}

void runTask3() {
    std::cout << "\n--- Task 3: OpenMP Selection Sort ---\n"; // Вывод заголовка / Header
    int sizes[] = {1000, 10000};          // Размеры массивов для теста / Test sizes
    
    for (int size : sizes) {              // Цикл по разным размерам / Loop sizes
        std::cout << "Processing array size: " << size << "...\n"; // Инфо о процессе / Status info
        std::vector<int> arr = generateRandomArray(size); // Генерация данных / Generate data
        std::vector<int> arr_copy = arr;  // Копия для второго теста / Copy for 2nd test

        double timeSeq = measureTime([&]() { selectionSortSequential(arr); }); // Замер посл. / Time seq
        double timePar = measureTime([&]() { selectionSortParallelImproved(arr_copy); }); // Замер парал. / Time par

        std::cout << "  Sequential Time: " << timeSeq << " ms\n"; // Вывод времени / Print time
        std::cout << "  Parallel Time:   " << timePar << " ms\n"; // Вывод времени / Print time
        std::cout << "  Speedup:         " << timeSeq / timePar << "x\n"; // Вывод ускорения / Print speedup
    }
}


// =================================================================================================
// ЗАДАЧА 4. СОРТИРОВКА НА GPU С ИСПОЛЬЗОВАНИЕМ CUDA
// =================================================================================================

#ifdef __CUDACC__ // Проверка, компилируется ли код компилятором NVCC / Check if compiling with NVCC
#include <cuda_runtime.h>            // Подключение Runtime API CUDA / Include CUDA Runtime
#include <device_launch_parameters.h> // Подключение параметров запуска ядер / Include launch params

// CUDA Kernel: Merge two sorted subarrays / Ядро CUDA: Слияние двух подмассивов
__global__ void mergeKernel(int* arr, int* temp, int width, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Вычисление глобального индекса потока / Calculate global thread ID
    int start = idx * 2 * width;      // Вычисление начала обрабатываемого блока / Calculate start index
    
    if (start < n) {                  // Проверка границ массива / Check bounds
        int mid = min(start + width, n);      // Индекс середины (начала второго подмассива) / Mid index
        int end = min(start + 2 * width, n);  // Индекс конца блока / End index
        
        int i = start;                // Индекс для левой части / Index for left part
        int j = mid;                  // Индекс для правой части / Index for right part
        int k = start;                // Индекс для временного массива / Index for temp array
        
        // Merge into temp array / Слияние во временный массив
        while (i < mid && j < end) {  // Пока есть элементы в обоих частях / While elements exist
            if (arr[i] <= arr[j]) {   // Если элемент слева меньше / If left smaller
                temp[k++] = arr[i++]; // Берем слева / Take from left
            } else {
                temp[k++] = arr[j++]; // Берем справа / Take from right
            }
        }
        while (i < mid) temp[k++] = arr[i++]; // Докопирование остатков левой части / Copy remaining left
        while (j < end) temp[k++] = arr[j++]; // Докопирование остатков правой части / Copy remaining right
        
        // Copy back to original array / Копирование обратно в исходный массив
        for (int x = start; x < end; x++) { // Цикл по обработанному участку / Loop processed chunk
            arr[x] = temp[x];         // Копирование / Copy
        }
    }
}

void mergeSortCUDA(std::vector<int>& h_arr) {
    int n = h_arr.size();      // Размер массива / Array size
    int* d_arr;                // Указатель на массив в памяти GPU / Pointer to GPU array
    int* d_temp;               // Указатель на временный массив GPU / Pointer to GPU temp array
    
    cudaMalloc(&d_arr, n * sizeof(int)); // Выделение памяти на GPU / Allocate GPU memory
    cudaMalloc(&d_temp, n * sizeof(int)); // Выделение памяти на GPU / Allocate GPU memory
    
    // Копирование данных с Host (CPU) на Device (GPU) / Copy data Host -> Device
    cudaMemcpy(d_arr, h_arr.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256; // Количество потоков в блоке / Threads per block
    
    // Bottom-Up Merge Sort Loop / Цикл сортировки слиянием снизу-вверх
    for (int width = 1; width < n; width *= 2) {
        // Вычисление необходимого количества блоков запуска / Calculate grid size
        int numMerges = (n + (2 * width) - 1) / (2 * width);
        int blocks = (numMerges + threadsPerBlock - 1) / threadsPerBlock;
        
        // Запуск ядра Merge / Launch Merge Kernel
        mergeKernel<<<blocks, threadsPerBlock>>>(d_arr, d_temp, width, n);
        cudaDeviceSynchronize(); // Ожидание завершения GPU / Wait for GPU finish
    }
    
    // Копирование результата обратно на Host / Copy result Device -> Host
    cudaMemcpy(h_arr.data(), d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_arr);  // Освобождение памяти GPU / Free GPU memory
    cudaFree(d_temp); // Освобождение памяти GPU / Free GPU memory
}

void runTask4() {
    std::cout << "\n--- Task 4: CUDA Merge Sort ---\n"; // Заголовок задачи / Task header
    int sizes[] = {10000, 100000};    // Размеры массивов / Array sizes
    
    for (int size : sizes) {          // Цикл по размерам / Loop sizes
        std::cout << "Processing array size: " << size << "...\n"; // Инфо / Info
        std::vector<int> arr = generateRandomArray(size); // Генерация / Generate
        
        double time = measureTime([&]() { mergeSortCUDA(arr); }); // Замер времени CUDA / Time CUDA
        
        std::cout << "  CUDA Implementation Time: " << time << " ms\n"; // Результат / Print result
    }
}

#else // Если компилируется не NVCC (обычный C++) / If not NVCC (standard C++)

// Fallback logic / Заглушка
void runTask4() {
    std::cout << "\n--- Task 4: CUDA Merge Sort ---\n";
    // Сообщение о пропуске / Skip message
    std::cout << "Skipping: CUDA compiler (nvcc) not detected or macro __CUDACC__ not defined.\n";
    std::cout << "The code for Task 4 is present in main.cpp but disabled.\n";
}

#endif


int main() {
    srand(time(0)); // Инициализация генератора случайных чисел / Seed RNG
    
    std::cout << "Starting Assignment 2 Solutions...\n"; // Приветствие / Start msg
    
    // Task 1 is theoretical / Задача 1 теоретическая
    std::cout << "Task 1: Theoretical answers are provided in the source code comments.\n";

    runTask2(); // Запуск задачи 2 / Run Task 2
    runTask3(); // Запуск задачи 3 / Run Task 3
    runTask4(); // Запуск задачи 4 / Run Task 4
    
    return 0; // Завершение программы / Exit
}
