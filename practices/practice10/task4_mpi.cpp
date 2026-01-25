#include <mpi.h>      // Подключение заголовочного файла MPI
#include <iostream>   // Ввод-вывод
#include <vector>     // Вектор
#include <numeric>    // Числовые алгоритмы
#include <algorithm>  // Алгоритмы (std::fill)

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // Инициализация среды MPI. Должна быть вызвана первой.

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Получение ранга (номера) текущего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Получение общего количества процессов

    // Режим масштабирования (передается через аргументы командной строки):
    // 0: Strong Scaling (Фиксированный общий размер задачи)
    // 1: Weak Scaling (Фиксированный размер задачи на каждый процесс)
    int mode = 0; 
    if (argc > 1) { // Проверяем, есть ли аргумент
        mode = std::atoi(argv[1]); // Конвертируем аргумент в число
    }
    
    // Total elements for Strong Scaling: 100 Million
    // Elements per process for Weak Scaling: 25 Million
    long long n_elements; // Количество элементов для ЭТОГО процесса
    
    if (mode == 0) {
        // Strong Scaling: Задача фиксированного размера делится между процессами
        long long total_elements = 100000000; // 100 млн всего
        n_elements = total_elements / size;   // Доля каждого процесса уменьшается с ростом size
    } else {
        // Weak Scaling: Каждому процессу дается фиксированный объем работы
        n_elements = 25000000; // 25 млн на процесс (общий размер растет с ростом size)
    }

    if (rank == 0) { // Только главный процесс (rank 0) выводит заголовок
        std::cout << "Task 4: MPI Scalability Analysis" << std::endl;
        std::cout << "Processes: " << size << std::endl;
        std::cout << "Elements per process: " << n_elements << std::endl;
        std::cout << "Mode: " << (mode == 0 ? "Strong Scaling" : "Weak Scaling") << std::endl;
    }

    // Выделение памяти и заполнение данных
    std::vector<double> data(n_elements); // Создание вектора
    // Заполняем единицами (1.0)
    std::fill(data.begin(), data.end(), 1.0);

    MPI_Barrier(MPI_COMM_WORLD); // Барьер: ждем, пока все процессы дойдут сюда (для честного старта таймера)
    double start_time = MPI_Wtime(); // Засекаем глобальное время начала

    // Локальные вычисления (Сумма элементов вектора)
    double local_sum = 0.0;
    for (const auto& val : data) {
        local_sum += val; // Простое суммирование
    }

    // Замеряем время окончания вычислений (до коммуникации)
    double compute_end = MPI_Wtime();

    // Редукция (Сбор результатов со всех процессов)
    double global_sum = 0.0;
    // MPI_Reduce берет local_sum каждого процесса, выполняет MPI_SUM и кладет результат в global_sum на процессе 0
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime(); // Общее время окончания (после коммуникации)

    if (rank == 0) { // Только 0-й процесс выводит результаты
        double total_time = end_time - start_time;         // Полное время
        double compute_time = compute_end - start_time;    // Время вычислений
        double comm_time = end_time - compute_end;         // Приблизительное время коммуникации
        
        std::cout << "Total Time: " << total_time << " s" << std::endl;
        std::cout << "Compute Time (Rank 0): " << compute_time << " s" << std::endl;
        std::cout << "Comm/Reduce Time: " << comm_time << " s" << std::endl;
        // Проверка результата: сумма всех 1.0 должна быть равна общему кол-ву элементов
        // Для Strong: 100млн. Для Weak: 25млн * size.
        // Эквивалентно n_elements (локальное) * size (для обоих случаев, так как в strong n_elements = total/size)
        // Поправка: в Strong n_elements = total/size. Sum = n_elements * size = total.
        // В Weak n_elements = const. Sum = n_elements * size. Верно.
        std::cout << "Global Sum: " << global_sum << " (Expected: " << (double)n_elements * size << ")" << std::endl;
        std::cout << "------------------------------------------" << std::endl; // Разделитель
    }

    MPI_Finalize(); // Завершение работы с MPI. Обязательно в конце.
    return 0;
}
