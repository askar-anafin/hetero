@echo off
setlocal

echo ==========================================
echo Setting up Environment
echo ==========================================

rem 1. Setup Visual Studio Environment (for cl.exe used by nvcc)
where cl >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] cl.exe not found. Searching for vcvars64.bat...
    if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
        echo [INFO] VS 2022/2019 environment initialized.
    ) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
        echo [INFO] VS 2019 environment initialized.
    ) else (
         echo [WARNING] vcvars64.bat not found. Compilation might fail if cl.exe is strictly required.
    )
)

rem 2. Setup MS-MPI SDK Paths
if "%MSMPI_INC%"=="" (
    if exist "C:\Program Files (x86)\Microsoft SDKs\MPI\Include\mpi.h" (
        set "MSMPI_INC=C:\Program Files (x86)\Microsoft SDKs\MPI\Include\"
        set "MSMPI_LIB64=C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\"
        echo [INFO] MS-MPI SDK found at default location.
    ) else (
        echo [ERROR] MS-MPI SDK not found. Task 4 will fail to compile.
    )
)

rem 3. Setup mpiexec Path
set "MPIEXEC_CMD=mpiexec"
where mpiexec >nul 2>nul
if %errorlevel% neq 0 (
    if exist "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" (
        set "MPIEXEC_CMD="C:\Program Files\Microsoft MPI\Bin\mpiexec.exe""
        echo [INFO] mpiexec found at C:\Program Files\Microsoft MPI\Bin\mpiexec.exe
    ) else (
        echo [WARNING] mpiexec not found. Task 4 execution will fail.
    )
)

echo.
echo ==========================================
echo Compiling Tasks
echo ==========================================

echo [COMPILE] Task 1: Sum
nvcc -o task1_sum.exe task1_sum.cu
if %errorlevel% neq 0 echo [FAIL] Task 1

echo [COMPILE] Task 2: Scan
nvcc -o task2_scan.exe task2_scan.cu
if %errorlevel% neq 0 echo [FAIL] Task 2

echo [COMPILE] Task 3: Hybrid
nvcc -o task3_hybrid.exe task3_hybrid.cu
if %errorlevel% neq 0 echo [FAIL] Task 3

echo [COMPILE] Task 4: MPI
nvcc -o task4_mpi.exe task4_mpi.cpp -I"%MSMPI_INC%\" -L"%MSMPI_LIB64%\" -lmsmpi
if %errorlevel% neq 0 echo [FAIL] Task 4

echo.
echo ==========================================
echo Running Tasks
echo ==========================================

if exist task1_sum.exe (
    echo.
    echo --- Running Task 1 ---
    task1_sum.exe
)

if exist task2_scan.exe (
    echo.
    echo --- Running Task 2 ---
    task2_scan.exe
)

if exist task3_hybrid.exe (
    echo.
    echo --- Running Task 3 ---
    task3_hybrid.exe
)

if exist task4_mpi.exe (
    echo.
    echo --- Running Task 4 ^(Benchmark^) ---
    rem Run for 2, 4, 8 processes as required
    for %%P in (2 4 8) do (
        echo Running with %%P processes...
        %MPIEXEC_CMD% -n %%P task4_mpi.exe
    )
)

echo.
echo ==========================================
echo All Tasks Completed
echo ==========================================
endlocal
pause
