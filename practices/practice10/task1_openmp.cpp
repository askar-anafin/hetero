#include <iostream> // Подключение библиотеки ввода-вывода
#include <vector>   // Подключение библиотеки для использования вектора
#include <numeric>  // Подключение библиотеки для числовых операций
#include <omp.h>    // Подключение библиотеки OpenMP
#include <cmath>    // Подключение математической библиотеки
#include <iomanip>  // Подключение библиотеки для манипуляции выводом

// Функция для обработки данных с заданным количеством потоков
// Возвращает время выполнения в секундах
double process_data(const std::vector<double>& data, int num_threads) {
    double start_time = omp_get_wtime(); // Запоминаем время начала выполнения

    double sum = 0.0;       // Переменная для суммы
    double mean = 0.0;      // Переменная для среднего значения
    double variance = 0.0;  // Переменная для дисперсии
    size_t n = data.size(); // Получаем размер вектора

    // Устанавливаем количество потоков для следующей параллельной области
    omp_set_num_threads(num_threads); 

    // 1. Вычисление Суммы
    // Директива для распараллеливания цикла for с редукцией переменной sum
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < n; ++i) { // Цикл по всем элементам вектора
        sum += data[i]; // Добавляем элемент к сумме
    }

    mean = sum / n; // Вычисляем среднее значение

    // 2. Вычисление Дисперсии
    // Дисперсия = sum((x_i - mean)^2) / n
    double sq_diff_sum = 0.0; // Переменная для суммы квадратов разностей
    // Директива для распараллеливания цикла for с редукцией переменной sq_diff_sum
    #pragma omp parallel for reduction(+:sq_diff_sum)
    for (size_t i = 0; i < n; ++i) { // Цикл по всем элементам
        double diff = data[i] - mean; // Вычисляем разность элемента и среднего
        sq_diff_sum += diff * diff;   // Добавляем квадрат разности к сумме
    }
    variance = sq_diff_sum / n; // Вычисляем дисперсию генеральной совокупности

    double end_time = omp_get_wtime(); // Запоминаем время окончания
    return end_time - start_time;      // Возвращаем затраченное время
}

int main() {
    const size_t DATA_SIZE = 50000000; // Размер данных (50 миллионов элементов)
    std::cout << "Initializing data (" << DATA_SIZE << " elements)..." << std::endl; // Вывод сообщения об инициализации
    
    std::vector<double> data(DATA_SIZE); // Создание вектора заданного размера
    
    // Инициализация данных последовательно (вне замера времени)
    for (size_t i = 0; i < DATA_SIZE; ++i) { // Цикл заполнения вектора
        data[i] = static_cast<double>(i) * 0.001; // Заполнение значениями
    }

    std::cout << "Starting performance analysis..." << std::endl; // Вывод сообщения о начале анализа
    std::cout << "--------------------------------------------------------------------------------------------------" << std::endl; // Разделитель
    // Вывод заголовка таблицы результатов
    std::cout << std::setw(10) << "Threads" 
              << std::setw(15) << "Time (s)" 
              << std::setw(15) << "Speedup" 
              << std::setw(15) << "Efficiency" 
              << std::setw(20) << "Est. Parallel Part" 
              << std::endl;
    std::cout << "--------------------------------------------------------------------------------------------------" << std::endl; // Разделитель

    // Тестирование с различным количеством потоков
    std::vector<int> thread_counts = {1, 2, 4, 8, 16}; // Вектор с количеством потоков
    
    // Разогрев (опционально)
    // process_data(data, 1); // Закомментировано для экономии времени

    double t1 = 0.0; // Переменная для хранения времени выполнения на 1 потоке

    for (int t : thread_counts) { // Цикл по всем вариантам количества потоков
        double time_taken = process_data(data, t); // Запуск обработки и получение времени
        
        if (t == 1) { // Если это запуск на 1 потоке
            t1 = time_taken; // Сохраняем время как базовое (T1)
        }

        double speedup = t1 / time_taken; // Вычисляем ускорение (S = T1 / Tn)
        double efficiency = speedup / t;  // Вычисляем эффективность (E = S / n)
        
        // Оценка параллельной части по закону Амдала: S = 1 / ((1-P) + P/N)
        // => (1-P) + P/N = 1/S
        // => 1 - P + P/N = 1/S
        // => P(1/N - 1) = 1/S - 1
        // => P = (1/S - 1) / (1/N - 1)
        double parallel_part = 0.0; // Переменная для доли параллельной части
        if (t > 1 && speedup > 1.0) { // Если потоков > 1 и есть ускорение
             parallel_part = (1.0/speedup - 1.0) / (1.0/(double)t - 1.0); // Вычисляем P
        } else if (t == 1) { // Для 1 потока
             parallel_part = 1.0; // Условно 100% (или не применимо)
        }

        // Вывод результатов в таблицу
        std::cout << std::setw(10) << t 
                  << std::setw(15) << std::fixed << std::setprecision(5) << time_taken
                  << std::setw(15) << speedup
                  << std::setw(15) << efficiency
                  << std::setw(20) << parallel_part
                  << std::endl;
    }
    
    return 0; // Завершение программы
}
