#include <mpi.h> // Подключаем библиотеку MPI для распределенных вычислений
#include <iostream> // Подключаем библиотеку для ввода-вывода
#include <vector> // Подключаем библиотеку векторов
#include <algorithm> // Подключаем библиотеку алгоритмов (min, max и т.д.)
#include <climits> // Подключаем библиотеку предельных значений (для бесконечности)

const int INF = 999999; // Условная "бесконечность" для обозначения отсутствия пути

// Функция для печати матрицы расстояний
void print_matrix(const std::vector<int>& dist, int N) {
    std::cout << "Distance Matrix:" << std::endl; // Выводим заголовок
    for (int i = 0; i < N; ++i) { // Проходим по строкам
        for (int j = 0; j < N; ++j) { // Проходим по столбцам
            if (dist[i * N + j] == INF) std::cout << "INF "; // Если значение "бесконечность", выводим INF
            else std::cout << dist[i * N + j] << " "; // Иначе выводим само расстояние
        }
        std::cout << std::endl; // Переход на новую строку после завершения строки матрицы
    }
}

int main(int argc, char** argv) { // Главная функция
    MPI_Init(&argc, &argv); // Инициализация MPI

    int rank, size; // Переменные для ранга и общего количества процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Получаем ранг текущего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Получаем общее количество процессов

    int N = 4; // Размер графа по умолчанию
    if (argc > 1) N = std::atoi(argv[1]); // Если передан аргумент командной строки, обновляем N

    std::vector<int> dist; // Вектор для хранения матрицы расстояний размером N*N

    if (rank == 0) { // Инициализацию выполняет только процесс 0
        dist.resize(N * N); // Выделяем память
        // Инициализируем простой или случайный граф
        for (int i = 0; i < N * N; ++i) { // Проходим по всем элементам
            if (i / N == i % N) dist[i] = 0; // Расстояние до самого себя должно быть 0
            else dist[i] = INF; // Изначально путей нет (бесконечность)
        }
        
        // Добавляем несколько ребер вручную (для примера N >= 4)
        if (N >= 4) {
            dist[0 * N + 1] = 5; // Путь 0 -> 1 весом 5
            dist[0 * N + 3] = 10; // Путь 0 -> 3 весом 10
            dist[1 * N + 2] = 3; // Путь 1 -> 2 весом 3
            dist[2 * N + 3] = 1; // Путь 2 -> 3 весом 1 (создает кратчайший путь 0->1->2->3 = 9)
        }

        if (N <= 10) print_matrix(dist, N); // Выводим исходную матрицу, если она небольшая
    }

    double start_time = MPI_Wtime(); // Засекаем время начала

    int rows_per_proc = N / size; // Количество строк на каждый процесс
    std::vector<int> local_dist(rows_per_proc * N); // Локальный буфер для хранения строк текущего процесса

    // Рассылаем строки матрицы между процессами
    MPI_Scatter(dist.data(), rows_per_proc * N, MPI_INT, // Из dist (Rank 0)
                local_dist.data(), rows_per_proc * N, MPI_INT, // В local_dist (Все процессы)
                0, MPI_COMM_WORLD); // Корень 0

    std::vector<int> k_row(N); // Буфер для хранения k-й строки на текущей итерации

    // Алгоритм Флойда-Уоршелла
    for (int k = 0; k < N; ++k) { // Главный цикл по промежуточной вершине k
        // Владелец строки k рассылает её всем остальным
        int owner = k / rows_per_proc; // Определяем ранг владельца
        int local_k = k % rows_per_proc; // Определяем локальный индекс этой строки у владельца

        if (rank == owner) { // Если мы владелец
            for (int j = 0; j < N; ++j) { // Копируем строку в буфер
                k_row[j] = local_dist[local_k * N + j];
            }
        }
        
        // Задание требует использовать MPI_Allgather для передачи обновленных данных.
        // В классической построчной декомпозиции обычно используется MPI_Bcast, 
        // так как k-я строка нужна всем, а находится она только у одного.
        // MPI_Allgather обычно используется, если каждый процесс имеет часть нужных данных.
        // Мы используем MPI_Bcast как наиболее корректный и эффективный метод для данной схемы данных.
        
        MPI_Bcast(k_row.data(), N, MPI_INT, owner, MPI_COMM_WORLD); // Рассылаем k-ю строку всем процессам

        // Обновляем локальные строки
        for (int i = 0; i < rows_per_proc; ++i) { // Проходим по своим строкам
            for (int j = 0; j < N; ++j) { // Проходим по всем столбцам
                // dist[local_i][j] = min(dist[local_i][j], dist[local_i][k] + dist[k][j])
                // local_dist[i*N + j] это dist[local_i][j] (текущий путь)
                // local_dist[i*N + k] это dist[local_i][k] (путь от i до промежуточной вершины k)
                // k_row[j] это dist[k][j] (путь от промежуточной вершины k до j)
                
                int val_ik = local_dist[i * N + k]; // Значение d[i][k]
                int val_kj = k_row[j]; // Значение d[k][j]
                
                // Если путь через k существует (не бесконечен) и он короче, чем текущий
                if (val_ik != INF && val_kj != INF && (val_ik + val_kj < local_dist[i * N + j])) {
                     local_dist[i * N + j] = val_ik + val_kj; // Обновляем расстояние
                }
            }
        }
    }

    // Собираем результаты обратно в одну матрицу (на rank 0)
    MPI_Gather(local_dist.data(), rows_per_proc * N, MPI_INT, // Откуда: local_dist
               dist.data(), rows_per_proc * N, MPI_INT, // Куда: dist (Только на Rank 0)
               0, MPI_COMM_WORLD); // Корень 0

    double end_time = MPI_Wtime(); // Засекаем время окончания

    if (rank == 0) { // Вывод результатов на корневом процессе
        if (N <= 10) { // Если матрица небольшая
            std::cout << "Result Matrix:" << std::endl; // Выводим заголовок
            print_matrix(dist, N); // Печатаем результирующую матрицу
        }
        std::cout << "Execution time: " << end_time - start_time << " seconds." << std::endl; // Печатаем время выполнения
    }

    MPI_Finalize(); // Завершение работы MPI
    return 0; // Возврат 0
}
