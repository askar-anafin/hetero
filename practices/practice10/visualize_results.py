import subprocess # Модуль для запуска внешних процессов
import re           # Модуль для регулярных выражений (хотя здесь используется split)
import matplotlib.pyplot as plt # Библиотека для построения графиков
import sys          # Системный модуль
import os           # Модуль для работы с файловой системой

# Конфигурация
# Путь к MPI Executable (может отличаться на разных машинах)
MPI_EXEC_PATH = r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"
BUILD_DIR = "build" # Папка с собранными .exe файлами

def run_command(command):
    """Запускает команду и возвращает ее вывод (stdout) как строку."""
    try:
        # Запуск процесса с перехватом stdout и stderr
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        if result.returncode != 0: # Если код возврата не 0 (ошибка)
            print(f"Error running command: {' '.join(command)}")
            print(result.stderr) # Печать ошибки
            return ""
        return result.stdout # Возврат успешного вывода
    except Exception as e:
        print(f"Exception running command: {e}")
        return ""

def parse_openmp_results():
    print("Running OpenMP Benchmark...") # Запуск бенчмарка OpenMP
    exe_path = os.path.join(BUILD_DIR, "task1_openmp.exe")
    if not os.path.exists(exe_path): # Проверка наличия файла
        print(f"Executable not found: {exe_path}")
        return None

    output = run_command([exe_path]) # Запуск
    
    threads = []      # Список для кол-ва потоков
    times = []        # Список времен
    speedups = []     # Список ускорений
    efficiencies = [] # Список эффективностей
    
    # Парсинг строк таблицы из вывода программы:
    #    Threads       Time (s)        Speedup     Efficiency  Est. Parallel Part
    #          1        0.07800        1.00000        1.00000             1.00000
    for line in output.splitlines():
        parts = line.split() # Разбиение строки на слова
        if not parts: continue # Пропуск пустых строк
        
        # Проверяем, начинается ли строка с числа (это строка данных)
        if parts[0].isdigit():
            try:
                t = int(parts[0])       # Потоки
                time = float(parts[1])  # Время
                speedup = float(parts[2]) # Ускорение
                eff = float(parts[3])   # Эффективность
                
                threads.append(t)
                times.append(time)
                speedups.append(speedup)
                efficiencies.append(eff)
            except ValueError:
                continue
                
    return threads, times, speedups, efficiencies

def parse_cuda_results():
    print("Running CUDA Memory Benchmark...") # Запуск CUDA теста
    exe_path = os.path.join(BUILD_DIR, "task2_memory.exe")
    if not os.path.exists(exe_path):
        print(f"Executable not found: {exe_path}")
        return None
        
    output = run_command([exe_path]) # Запуск
    
    labels = [] # Названия тестов
    times = []  # Времена
    
    # Парсим строки вида: "Kernel: Coalesced | Time: 1.14448 ms"
    
    for line in output.splitlines():
        if "Kernel:" in line and "Time:" in line:
            try:
                # Извлекаем имя ядра
                name_part = line.split("|")[0].split(":")[1].strip()
                # Извлекаем время
                time_part = line.split("|")[1].split(":")[1].strip().split()[0]
                
                labels.append(name_part)
                times.append(float(time_part))
            except Exception as e:
                print(f"Error parsing line '{line}': {e}")
                
    return labels, times

def parse_hybrid_results():
    print("Running Hybrid CPU+GPU Benchmark...") # Запуск гибридного теста
    exe_path = os.path.join(BUILD_DIR, "task3_hybrid.exe")
    if not os.path.exists(exe_path):
        print(f"Executable not found: {exe_path}")
        return None
        
    output = run_command([exe_path]) # Выполнение
    
    modes = [] # Список режимов (Synchronous/Hybrid)
    times = [] # Список времен выполнения
    
    # Парсим строки вывода:
    # Mode: Synchronous | Time: 45.12 ms
    # Total Hybrid Time: 24.03 ms 
    
    for line in output.splitlines():
        if "Mode: Synchronous" in line: # Если это синхронный режим
            modes.append("Synchronous")
            times.append(float(line.split("Time:")[1].replace("ms","").strip()))
        elif "Total Hybrid Time:" in line: # Если это гибридный режим
            modes.append("Hybrid (Async)")
            times.append(float(line.split("Time:")[1].replace("ms","").strip()))
            
    return modes, times

def parse_mpi_results():
    print("Running MPI Scaling Benchmarks...") # Запуск MPI тестов
    exe_path = os.path.join(BUILD_DIR, "task4_mpi.exe")
    if not os.path.exists(exe_path):
        print(f"Executable not found: {exe_path}")
        return None

    # Мы будем запускать для 1, 2 и 4 процессов для обоих режимов
    procs = [1, 2, 4]
    strong_times = [] # Времена для Strong Scaling
    weak_times = []   # Времена для Weak Scaling
    
    # Mode 0: Strong Scaling (Фиксированный общий размер)
    print("  Measuring Strong Scaling...")
    for p in procs:
        cmd = [MPI_EXEC_PATH, "-n", str(p), exe_path, "0"] # Команда mpiexec
        out = run_command(cmd)
        
        # Ищем строку: "Total Time: 0.0759979 s"
        found = False
        for line in out.splitlines():
            if "Total Time:" in line:
                t = float(line.split(":")[1].replace("s", "").strip())
                strong_times.append(t)
                found = True
                break
        if not found:
            strong_times.append(0)

    # Mode 1: Weak Scaling (Фиксированный размер на процесс)
    print("  Measuring Weak Scaling...")
    for p in procs:
        cmd = [MPI_EXEC_PATH, "-n", str(p), exe_path, "1"]
        out = run_command(cmd)
        
        found = False
        for line in out.splitlines():
            if "Total Time:" in line:
                t = float(line.split(":")[1].replace("s", "").strip())
                weak_times.append(t)
                found = True
                break
        if not found:
            weak_times.append(0)
            
    return procs, strong_times, weak_times

def main():
    # 1. Сбор данных OpenMP
    omp_data = parse_openmp_results()
    
    # 2. Сбор данных CUDA
    cuda_data = parse_cuda_results()
    
    # 3. Сбор данных Hybrid
    hybrid_data = parse_hybrid_results()

    # 4. Сбор данных MPI
    mpi_data = parse_mpi_results()
    
    # Настройка графиков
    plt.style.use('bmh') # Стиль графика
    fig = plt.figure(figsize=(15, 12)) # Размер окна (немного выше)
    fig.suptitle('Heterogeneous Computing Benchmark Results', fontsize=16) # Общий заголовок

    # --- График 1: Ускорение OpenMP ---
    ax1 = fig.add_subplot(2, 2, 1) # Создание подграфика (2 строки, 2 колонки, номер 1)
    if omp_data and len(omp_data[0]) > 0:
        threads, times, speedups, effs = omp_data
        ax1.plot(threads, speedups, 'o-', label='Measured Speedup', linewidth=2) # Линия ускорения
        ax1.plot(threads, threads, 'k--', label='Ideal Linear Speedup', alpha=0.5) # Идеальная линия
        ax1.set_title('Task 1: OpenMP Speedup (CPU)')
        ax1.set_xlabel('Threads')
        ax1.set_ylabel('Speedup (T1/Tn)')
        ax1.set_xticks(threads)
        ax1.grid(True)
        
        # Добавление эффективности на правую ось Y
        ax1b = ax1.twinx()
        ax1b.plot(threads, effs, 'r^:', label='Efficiency', alpha=0.7)
        ax1b.set_ylabel('Efficiency', color='r')
        ax1b.tick_params(axis='y', labelcolor='r')

        # Merging legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    else:
        ax1.text(0.5, 0.5, "No OpenMP Data Found", ha='center', va='center')

    # --- График 2: Задержка памяти CUDA ---
    ax2 = fig.add_subplot(2, 2, 2)
    if cuda_data and len(cuda_data[0]) > 0:
        labels, times = cuda_data
        
        # Укорачиваем подписи для красоты
        short_labels = []
        for l in labels:
            if "Stride 1009" in l: short_labels.append("Stride(1009)")
            elif "Stride 32" in l: short_labels.append("Stride(32)")
            elif "Coalesced" in l: short_labels.append("Coalesced")
            elif "Shared" in l: short_labels.append("Shared Mem")
            else: short_labels.append(l)

        # Столбчатая диаграмма
        bars = ax2.bar(short_labels, times, color=['blue', 'green', 'red', 'orange'])
        ax2.set_title('Task 2: GPU Kernel Execution Time')
        ax2.set_ylabel('Time (ms) [Lower is Better]')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Добавляем подписи значений над столбцами
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
    else:
        ax2.text(0.5, 0.5, "No CUDA Data Found", ha='center', va='center')

    # --- График 3: Hybrid CPU+GPU ---
    # Переместим MPI на 4-й слот, а Hybrid на 3-й
    ax3 = fig.add_subplot(2, 2, 3)
    if hybrid_data and len(hybrid_data[0]) > 0:
        modes, times = hybrid_data
        
        bars = ax3.bar(modes, times, color=['gray', 'purple'])
        ax3.set_title('Task 3: Hybrid vs Serial Performance')
        ax3.set_ylabel('Total Time (ms) [Lower is Better]')
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
                    
        # Вычисление ускорения
        if len(times) == 2 and times[1] > 0:
             speedup = times[0] / times[1]
             ax3.text(0.5, 0.9, f"Overlap Speedup: {speedup:.2f}x", 
                      transform=ax3.transAxes, ha='center', fontsize=12, 
                      bbox=dict(facecolor='white', alpha=0.8))
    else:
        ax3.text(0.5, 0.5, "No Hybrid Data Found", ha='center', va='center')


    # --- График 4: Масштабируемость MPI ---
    ax4 = fig.add_subplot(2, 2, 4)
    if mpi_data and len(mpi_data[0]) > 0:
        procs, strong_t, weak_t = mpi_data
        
        # Линия Strong Scaling (должна падать)
        ax4.plot(procs, strong_t, 'bo-', label='Strong Scaling (Time)', linewidth=2)
        ax4.set_xlabel('Process Count')
        ax4.set_ylabel('Time (s)', color='b')
        ax4.tick_params(axis='y', labelcolor='b')
        ax4.set_xticks(procs)
        
        # Линия Weak Scaling (должна быть прямой)
        ax4.plot(procs, weak_t, 'rs-', label='Weak Scaling (Time)', linewidth=2)
        
        ax4.set_title('Task 4: MPI Scalability')
        ax4.legend()
        ax4.grid(True)
    else:
        ax4.text(0.5, 0.5, "No MPI Data Found", ha='center', va='center')

    plt.tight_layout() # Автоматическая подгонка отступов
    output_file = 'benchmark_results.png'
    plt.savefig(output_file, dpi=100) # Сохранение изображения
    print(f"Generated plot: {output_file}")
    plt.show() # Показ окна

if __name__ == "__main__":
    main()
