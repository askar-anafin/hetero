#ifndef UTILS_H
#define UTILS_H

#include <iostream> // Подключение библиотеки ввода-вывода
#include <vector> // Подключение библиотеки векторов (динамических массивов)
#include <algorithm> // Подключение библиотеки алгоритмов (например, std::sort)
#include <random> // Подключение библиотеки для генерации случайных чисел
#include <chrono> // Подключение библиотеки для работы со временем
#include <iomanip> // Подключение библиотеки для форматирования вывода
#include <cmath> // Подключение математической библиотеки

// Класс таймера для точного измерения времени
class CpuTimer {
    using Clock = std::chrono::high_resolution_clock; // Использование таймера высокого разрешения
    std::chrono::time_point<Clock> start_time; // Переменная для хранения времени начала
public:
    // Метод запуска таймера
    void start() {
        start_time = Clock::now(); // Запоминаем текущее время
    }
    
    // Метод остановки таймера и получения прошедшего времени в миллисекундах
    double stop() {
        auto end_time = Clock::now(); // Запоминаем время окончания
        return std::chrono::duration<double, std::milli>(end_time - start_time).count(); // Возвращаем разницу в миллисекундах
    }
};

// Функция заполнения массива случайными числами float
void fillArray(float* arr, size_t size) {
    std::random_device rd; // Инициализация источника случайности (аппаратного, если доступен)
    std::mt19937 gen(rd()); // Инициализация генератора Мерсенна Твистера
    std::uniform_real_distribution<float> dis(0.0f, 1000.0f); // Равномерное распределение от 0.0 до 1000.0

    for (size_t i = 0; i < size; ++i) { // Цикл по всем элементам массива
        arr[i] = dis(gen); // Присваивание случайного значения элементу
    }
}

// Функция проверки, отсортирован ли массив
bool isSorted(const float* arr, size_t size) {
    for (size_t i = 0; i < size - 1; ++i) { // Проход по массиву до предпоследнего элемента
        if (arr[i] > arr[i + 1]) { // Если текущий элемент больше следующего
            return false; // Массив не отсортирован
        }
    }
    return true; // Массив отсортирован
}

// Функция проверки результатов сравнением с эталонной отсортированной копией
bool verifyResults(const float* arr, const float* ref, size_t size) {
    for(size_t i = 0; i < size; ++i) { // Проход по всем элементам
        if (std::abs(arr[i] - ref[i]) > 1e-4f) { // Сравнение чисел с плавающей точкой с учетом погрешности
            return false; // Если разница больше допустимой погрешности, возвращаем false
        }
    }
    return true; // Результаты совпадают
}

// Функция печати заголовка таблицы
void printTableHeader() {
    std::cout << std::left << std::setw(15) << "Size"  // Печать заголовка "Size" с шириной 15
              << std::setw(20) << "CPU Seq (ms)"  // Печать заголовка "CPU Seq (ms)" с шириной 20
              << std::setw(20) << "CPU OMP (ms)"  // Печать заголовка "CPU OMP (ms)" с шириной 20
              << std::setw(20) << "GPU Bitonic (ms)"  // Печать заголовка "GPU Bitonic (ms)" с шириной 20
              << "Result" << std::endl; // Печать заголовка "Result" и перевод строки
    std::cout << std::string(90, '-') << std::endl; // Печать разделительной линии
}

// Функция печати строки таблицы с результатами
void printTableRow(size_t size, double cpuSeqTime, double cpuOmpTime, double gpuTime, bool passed) {
    std::cout << std::left << std::setw(15) << size  // Вывод размера массива
              << std::setw(20) << std::fixed << std::setprecision(3) << cpuSeqTime  // Вывод времени CPU Seq
              << std::setw(20) << std::fixed << std::setprecision(3) << cpuOmpTime  // Вывод времени CPU OMP
              << std::setw(20) << std::fixed << std::setprecision(3) << gpuTime  // Вывод времени GPU
              << (passed ? "PASS" : "FAIL") << std::endl; // Вывод результата (PASS или FAIL)
}

#endif // UTILS_H
