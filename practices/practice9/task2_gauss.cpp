#include <mpi.h> // Подключаем библиотеку MPI для распределенных вычислений
#include <iostream> // Подключаем библиотеку для ввода-вывода
#include <vector> // Подключаем библиотеку для использования динамических массивов (векторов)
#include <cmath> // Подключаем математическую библиотеку
#include <cstdlib> // Подключаем библиотеку стандартных функций (atoi, rand)

// Функция для вывода системы уравнений (Матрица A и вектор b)
void print_system(const std::vector<double>& A, const std::vector<double>& b, int N) {
    std::cout << "System:" << std::endl; // Выводим заголовок
    for (int i = 0; i < N; ++i) { // Проходим по строкам
        for (int j = 0; j < N; ++j) { // Проходим по столбцам
            std::cout << A[i * N + j] << " "; // Выводим элемент A[i][j]
        }
        std::cout << "| " << b[i] << std::endl; // Выводим элемент вектора b[i] и переходим на новую строку
    }
}

int main(int argc, char** argv) { // Главная функция программы
    MPI_Init(&argc, &argv); // Инициализация среды MPI

    int rank, size; // Переменные для ранга процесса и размера коммуникатора
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Получаем номер текущего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Получаем общее количество процессов

    int N = 4; // Размерность системы по умолчанию
    if (argc > 1) N = std::atoi(argv[1]); // Если передан аргумент командной строки, используем его как N

    std::vector<double> A; // Вектор для хранения матрицы A (размером N*N)
    std::vector<double> b; // Вектор для хранения правой части b (размером N)

    if (rank == 0) { // Только процесс с рангом 0 инициализирует данные
        A.resize(N * N); // Выделяем память под матрицу
        b.resize(N); // Выделяем память под вектор
        std::srand(42); // Инициализируем генератор случайных чисел с фиксированным зерном
        for (int i = 0; i < N * N; ++i) A[i] = (std::rand() % 10) + 1; // Заполняем матрицу случайными числами
        for (int i = 0; i < N; ++i) b[i] = (std::rand() % 10) + 1; // Заполняем вектор случайными числами
        
        // Обеспечиваем диагональное преобладание для численной устойчивости метода
        for (int i=0; i<N; ++i) A[i*N + i] += 20.0; // Увеличиваем диагональные элементы

        if (N <= 10) print_system(A, b, N); // Если система небольшая, выводим её на экран
    }

    double start_time = MPI_Wtime(); // Засекаем время начала вычислений

    // Проверка корректности распределения (для простоты считаем, что N делится на size)
    if (N % size != 0) { // Если N не делится на количество процессов нацело
        if (rank == 0) std::cerr << "Warning: N(" << N << ") is not divisible by size(" << size << "). Logic might fail or trail." << std::endl; // Выводим предупреждение
        // В рамках данной задачи предполагаем, что запуск будет с корректными параметрами
    }

    int rows_per_proc = N / size; // Количество строк, обрабатываемых одним процессом
    std::vector<double> local_A(rows_per_proc * N); // Локальная часть матрицы A
    std::vector<double> local_b(rows_per_proc); // Локальная часть вектора b

    // Распределяем строки матрицы A между всеми процессами
    MPI_Scatter(A.data(), rows_per_proc * N, MPI_DOUBLE, // Данные на отправку (с rank 0)
                local_A.data(), rows_per_proc * N, MPI_DOUBLE, // Буфер приема
                0, MPI_COMM_WORLD); // Корневой процесс
    
    // Распределяем элементы вектора b между всеми процессами
    MPI_Scatter(b.data(), rows_per_proc, MPI_DOUBLE, // Данные на отправку
                local_b.data(), rows_per_proc, MPI_DOUBLE, // Буфер приема
                0, MPI_COMM_WORLD); // Корневой процесс

    // Прямой ход метода Гаусса (Forward Elimination)
    std::vector<double> pivot_row(N + 1); // Буфер для хранения ведущей строки (включая элемент вектора b)

    for (int k = 0; k < N; ++k) { // Цикл по всем шагам элиминации (по диагональным элементам)
        // Определяем, какой процесс владеет строкой k
        int owner = k / rows_per_proc; // Ранг владельца
        int local_k = k % rows_per_proc; // Локальный индекс строки на процессе-владельце

        if (rank == owner) { // Если текущий процесс владеет ведущей строкой
             for (int j = 0; j < N; ++j) {
                 pivot_row[j] = local_A[local_k * N + j]; // Копируем строку матрицы в буфер
             }
             pivot_row[N] = local_b[local_k]; // Копируем элемент вектора b
        }

        // Рассылаем ведущую строку всем процессам
        MPI_Bcast(pivot_row.data(), N + 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // Выполняем исключение элементов под главной диагональю
        for (int i = 0; i < rows_per_proc; ++i) { // Проходим по всем локальным строкам
            int global_row_idx = rank * rows_per_proc + i; // Вычисляем глобальный индекс строки
            if (global_row_idx > k) { // Обрабатываем только строки, расположенные ниже ведущей строки k
                double factor = local_A[i * N + k] / pivot_row[k]; // Вычисляем множитель для обнуления элемента
                for (int j = k; j < N; ++j) { // Вычитаем ведущую строку из текущей
                    local_A[i * N + j] -= factor * pivot_row[j]; // Операция A[i][j] = A[i][j] - factor * A[k][j]
                }
                local_b[i] -= factor * pivot_row[N]; // Аналогично обновляем правую часть: b[i] = b[i] - factor * b[k]
            }
        }
    }

    // Собираем результаты обратно на процесс 0 для обратного хода
    // Примечание: Параллельный обратный ход сложен, часто его делают последовательно на одном узле.
    MPI_Gather(local_A.data(), rows_per_proc * N, MPI_DOUBLE, // Отправляем локальные строки
               A.data(), rows_per_proc * N, MPI_DOUBLE, // Принимаем в общую матрицу (на rank 0)
               0, MPI_COMM_WORLD);

    MPI_Gather(local_b.data(), rows_per_proc, MPI_DOUBLE, // Отправляем локальные части b
               b.data(), rows_per_proc, MPI_DOUBLE, // Принимаем в общий вектор b (на rank 0)
               0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime(); // Засекаем время окончания прямого хода

    if (rank == 0) { // Только процесс 0 выполняет обратный ход
        // Обратный ход (Backward Substitution)
        std::vector<double> x(N); // Вектор решения
        for (int i = N - 1; i >= 0; --i) { // Идем от последней строки к первой
            double sum = 0; // Переменная для суммы известных членов
            for (int j = i + 1; j < N; ++j) { // Суммируем найденные значения x
                sum += A[i * N + j] * x[j];
            }
            x[i] = (b[i] - sum) / A[i * N + i]; // Выражаем x[i]
        }

        std::cout << "Solution x:" << std::endl; // Выводим заголовок решения
        for (int i = 0; i < N; ++i) { // Проходим по всем элементам решения
            std::cout << "x[" << i << "] = " << x[i] << std::endl; // Выводим x[i]
        }
        std::cout << "Execution time: " << end_time - start_time << " seconds." << std::endl; // Выводим время выполнения
    }

    MPI_Finalize(); // Завершаем работу MPI
    return 0; // Возвращаем код успеха
}
