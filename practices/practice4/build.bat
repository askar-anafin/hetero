@echo off
REM Build script for CUDA programs - initializes VS environment

echo Setting up Visual Studio 2022 environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo.
echo Compiling CUDA program...
nvcc -O3 -arch=sm_75 main.cu -o benchmark.exe -lcurand

if %errorlevel% == 0 (
    echo.
    echo ===================================
    echo Build successful!
    echo ===================================
    echo.
    echo Run with: benchmark.exe
    echo Or run with graphics: build.bat run
) else (
    echo.
    echo Build failed! Check errors above.
    exit /b 1
)

if "%1"=="run" (
    echo.
    echo Running benchmark...
    echo.
    benchmark.exe
    
    if exist results.csv (
        echo.
        echo Generating plots...
        python plot_results.py
    )
)
