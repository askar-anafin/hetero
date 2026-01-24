#include <iostream> // Подключение библиотеки ввода-вывода
#include <fstream> // Подключение библиотеки файлового ввода-вывода
#include <vector> // Подключение библиотеки векторов
#include <cmath> // Подключение математической библиотеки
#include <cuda_runtime.h> // Подключение CUDA Runtime
#include "utils.h" // Подключение вспомогательных функций
#include "cpu_sort.h" // Подключение функций сортировки CPU
#include "gpu_sort.cuh" // Подключение функций сортировки GPU

int main() {
    printTableHeader(); // Вывод заголовка таблицы результатов

    // Размеры массивов для теста: степени двойки от 2^16 до 2^24
    // 2^16 = 65,536
    // 2^20 = 1,048,576
    // 2^24 = 16,777,216
    std::vector<size_t> sizes; // Вектор для хранения размеров
    for (int i = 16; i <= 24; i += 2) { // Цикл по степеням двойки с шагом 2
        sizes.push_back(1 << i); // Добавление размера в вектор
    }

    std::ofstream outfile("results.csv"); // Открытие файла для записи результатов в CSV
    outfile << "Size,CPU_Seq,CPU_OMP,GPU_Bitonic" << std::endl; // Запись заголовка CSV

    for (size_t size : sizes) { // Цикл по всем выбранным размерам
        // Выделение памяти на хосте (CPU)
        float* h_arr = new float[size]; // Массив с исходными данными
        float* h_ref = new float[size]; // Массив для эталонной сортировки
        float* h_copy = new float[size]; // Массив для тестирования сортировок

        // Заполнение массива
        fillArray(h_arr, size); // Заполнение случайными числами
        std::copy(h_arr, h_arr + size, h_ref); // Копирование данных в эталонный массив

        // 1. Последовательная сортировка на CPU (используя std::sort на эталонном массиве)
        CpuTimer timer; // Создание таймера
        timer.start(); // Запуск таймера
        sequentialSort(h_ref, size); // Запуск последовательной сортировки
        double cpuSeqTime = timer.stop(); // Остановка таймера и получение времени

        // 2. Параллельная сортировка на CPU
        std::copy(h_arr, h_arr + size, h_copy); // Восстановление исходных данных из копии
        timer.start(); // Запуск таймера
        parallelSort(h_copy, size); // Запуск параллельной сортировки
        double cpuOmpTime = timer.stop(); // Остановка таймера и получение времени
        
        // Проверка корректности параллельной сортировки CPU
        bool ompPassed = verifyResults(h_copy, h_ref, size);
        if (!ompPassed) { // Если проверка не пройдена
             std::cerr << "CPU Parallel Sort Failed for size " << size << std::endl; // Вывод ошибки
        }

        // 3. Битоническая сортировка на GPU
        float* d_arr; // Указатель на память устройства (GPU)
        cudaMalloc(&d_arr, size * sizeof(float)); // Выделение памяти на GPU
        
        // Измерение времени (Передача данных + Сортировка + Возврат данных)
        timer.start(); // Запуск таймера
        cudaMemcpy(d_arr, h_arr, size * sizeof(float), cudaMemcpyHostToDevice); // Копирование данных с Host на Device
        gpuBitonicSort(d_arr, size); // Запуск ядра битонической сортировки
        cudaMemcpy(h_copy, d_arr, size * sizeof(float), cudaMemcpyDeviceToHost); // Копирование данных с Device на Host
        double gpuTime = timer.stop(); // Остановка таймера и получение времени

        cudaFree(d_arr); // Освобождение памяти на GPU
        
        // Проверка корректности сортировки GPU
        bool gpuPassed = verifyResults(h_copy, h_ref, size);
         if (!gpuPassed) { // Если проверка не пройдена
             std::cerr << "GPU Bitonic Sort Failed for size " << size << std::endl; // Вывод ошибки
        }

        // Вывод строки результата в консоль
        printTableRow(size, cpuSeqTime, cpuOmpTime, gpuTime, ompPassed && gpuPassed);
        // Запись результата в CSV файл
        outfile << size << "," << cpuSeqTime << "," << cpuOmpTime << "," << gpuTime << std::endl;

        // Освобождение памяти на хосте
        delete[] h_arr;
        delete[] h_ref;
        delete[] h_copy;
    }
    
    outfile.close(); // Закрытие CSV файла

    return 0; // Завершение программы
}
