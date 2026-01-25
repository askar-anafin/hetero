@echo off
setlocal

if not exist build mkdir build

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo ========================================
echo Building Task 1: OpenMP
echo ========================================
g++ -fopenmp -O2 -o build\task1_openmp.exe task1_openmp.cpp
if %errorlevel% neq 0 exit /b %errorlevel%


echo.
echo ========================================
echo Building Task 2: CUDA Memory Optimization
echo ========================================
nvcc -O3 -o build\task2_memory.exe task2_memory.cu
if %errorlevel% neq 0 exit /b %errorlevel%


echo.
echo ========================================
echo Building Task 3: Hybrid CPU+GPU
echo ========================================
nvcc -O3 -Xcompiler "/openmp" -o build\task3_hybrid.exe task3_hybrid.cu
if %errorlevel% neq 0 exit /b %errorlevel%


echo.
echo ========================================
echo Building Task 4: MPI Scalability Analysis
echo ========================================
cl /EHsc /O2 /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" /Fe:build\task4_mpi.exe task4_mpi.cpp /link /LIBPATH:"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" msmpi.lib
if %errorlevel% neq 0 (
    echo MS-MPI build failed. Check if MS-MPI SDK is installed.
)

echo.
echo ========================================
echo Running Task 1: OpenMP Analysis
echo ========================================
build\task1_openmp.exe

echo.
echo ========================================
echo Running Task 2: CUDA Memory Optimization
echo ========================================
build\task2_memory.exe

echo.
echo ========================================
echo Running Task 3: Hybrid CPU+GPU
echo ========================================
build\task3_hybrid.exe

echo.
echo ========================================
echo Running Task 4: MPI Scalability
echo ========================================
if exist build\task4_mpi.exe (
    echo [Strong Scaling - Fixed Total Size: 100M elements]
    for %%n in (1 2 4) do (
        echo ------------------------------------------
        echo Running with %%n processes...
        "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" -n %%n build\task4_mpi.exe 0
    )
    
    echo.
    echo [Weak Scaling - Fixed Size Per Process: 25M elements]
    for %%n in (1 2 4) do (
        echo ------------------------------------------
        echo Running with %%n processes...
        "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" -n %%n build\task4_mpi.exe 1
    )
)

echo.
echo ========================================
echo Generating Results Visualization
echo ========================================
python visualize_results.py

echo.
echo ========================================
echo All tasks executed.
echo ========================================
