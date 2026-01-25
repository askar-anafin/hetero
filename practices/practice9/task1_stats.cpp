#include <mpi.h> // Подключаем библиотеку MPI для распределенных вычислений
#include <iostream> // Подключаем библиотеку ввода-вывода
#include <vector> // Подключаем библиотеку для работы с вектором
#include <numeric> // Подключаем библиотеку для числовых операций
#include <cmath> // Подключаем библиотеку математических функций
#include <cstdlib> // Подключаем библиотеку стандартных утилит (rand, srand)
#include <ctime> // Подключаем библиотеку работы со временем

int main(int argc, char** argv) { // Основная функция программы
    MPI_Init(&argc, &argv); // Инициализация среды MPI

    int rank, size; // Переменные для ранга процесса и общего количества процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Получаем ранг текущего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Получаем общее количество процессов

    const int N = 1000000; // Объявляем общий размер массива данных
    std::vector<double> global_data; // Вектор для хранения всех данных (только на процессе root)
    
    // 1. Создаем массив случайных чисел на процессе с рангом 0
    if (rank == 0) { // Проверяем, является ли текущий процесс главным (root)
        global_data.resize(N); // Выделяем память под N элементов
        std::srand(static_cast<unsigned>(std::time(nullptr))); // Инициализируем генератор случайных чисел
        for (int i = 0; i < N; ++i) { // Цикл для заполнения массива
            global_data[i] = static_cast<double>(std::rand()) / RAND_MAX * 100.0; // Генерируем случайное число от 0 до 100
        }
    }

    // Измеряем время начала выполнения
    double start_time = MPI_Wtime(); // Засекаем текущее время

    // 2. Распределяем массив с помощью MPI_Scatterv (для обработки остатков)
    int base_count = N / size; // Базовое количество элементов на каждый процесс
    int remainder = N % size; // Остаток от деления (сколько процессов получат на 1 элемент больше)

    std::vector<int> sendcounts(size); // Вектор для хранения количества элементов, отправляемых каждому процессу
    std::vector<int> displs(size); // Вектор смещений (откуда начинать чтение для каждого процесса)

    if (rank == 0) { // Только главный процесс вычисляет схему распределения
        int offset = 0; // Переменная для текущего смещения
        for (int i = 0; i < size; ++i) { // Цикл по всем процессам
            sendcounts[i] = base_count + (i < remainder ? 1 : 0); // Если индекс меньше остатка, добавляем +1 элемент
            displs[i] = offset; // Записываем смещение для i-го процесса
            offset += sendcounts[i]; // Сдвигаем смещение на количество отправленных элементов
        }
    }

    // Широковещательная рассылка не требуется для sendcounts/displs в Scatterv, 
    // но каждый процесс должен знать, сколько элементов он получит.
    int local_count = base_count + (rank < remainder ? 1 : 0); // Вычисляем локальное количество элементов для текущего процесса
    
    std::vector<double> local_data(local_count); // Выделяем память под локальный буфер данных
    
    MPI_Scatterv(rank == 0 ? global_data.data() : nullptr, // Указатель на отправляемые данные (только на rank 0)
                 rank == 0 ? sendcounts.data() : nullptr, // Массив количества элементов для отправки (только на rank 0)
                 rank == 0 ? displs.data() : nullptr, // Массив смещений (только на rank 0)
                 MPI_DOUBLE, // Тип отправляемых данных
                 local_data.data(), // Указатель на буфер приема
                 local_count, // Количество элементов, которые примет текущий процесс
                 MPI_DOUBLE, // Тип принимаемых данных
                 0, // Ранг корневого процесса (отправителя)
                 MPI_COMM_WORLD); // Коммуникатор

    // 3. Вычисляем локальные суммы
    double local_sum = 0.0; // Переменная для локальной суммы элементов
    double local_sq_sum = 0.0; // Переменная для локальной суммы квадратов элементов
    
    for (double val : local_data) { // Проходим по всем полученным элементам
        local_sum += val; // Добавляем значение к сумме
        local_sq_sum += val * val; // Добавляем квадрат значения к сумме квадратов
    }

    // 4. Собираем результаты (редукция) на процессе с рангом 0
    double global_sum = 0.0; // Переменная для общей суммы (будет заполнена на root)
    double global_sq_sum = 0.0; // Переменная для общей суммы квадратов (будет заполнена на root)

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // Суммируем local_sum со всех процессов в global_sum
    MPI_Reduce(&local_sq_sum, &global_sq_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // Суммируем local_sq_sum со всех процессов в global_sq_sum

    double end_time = MPI_Wtime(); // Засекаем время окончания вычислений

    // 5. Вычисляем статистику на процессе с рангом 0
    if (rank == 0) { // Если текущий процесс - главный
        double mean = global_sum / N; // Вычисляем среднее значение
        // Формула дисперсии: E[X^2] - (E[X])^2
        // Или sum( (x - mean)^2 ) / N = (sum(x^2) - 2*mean*sum(x) + N*mean^2) / N
        // = sum(x^2)/N - 2*mean^2 + mean^2 = sum(x^2)/N - mean^2
        
        double variance = (global_sq_sum / N) - (mean * mean); // Вычисляем дисперсию через среднее квадратов и квадрат среднего
        double std_dev = std::sqrt(variance); // Вычисляем стандартное отклонение как корень из дисперсии

        std::cout << "N: " << N << std::endl; // Выводим общее количество элементов
        std::cout << "Mean: " << mean << std::endl; // Выводим среднее значение
        std::cout << "Standard Deviation: " << std_dev << std::endl; // Выводим стандартное отклонение
        std::cout << "Execution time: " << end_time - start_time << " seconds." << std::endl; // Выводим время выполнения
    }

    MPI_Finalize(); // Завершаем работу с MPI
    return 0; // Возвращаем 0, программа завершена успешно
}
