@echo off
REM Setup VS Environment specifically for the user's setup
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo ==========================================
echo Running Assignment 3 Tasks
echo ==========================================

echo [Task 1] Compiling...
nvcc -o task1 task1.cu
if %errorlevel% neq 0 (
    echo Compilation of Task 1 failed!
    exit /b %errorlevel%
)
echo [Task 1] Running...
task1

echo.
echo ------------------------------------------
echo.

echo [Task 2] Compiling...
nvcc -o task2 task2.cu
if %errorlevel% neq 0 (
    echo Compilation of Task 2 failed!
    exit /b %errorlevel%
)
echo [Task 2] Running...
task2

echo.
echo ------------------------------------------
echo.

echo [Task 3] Compiling...
nvcc -o task3 task3.cu
if %errorlevel% neq 0 (
    echo Compilation of Task 3 failed!
    exit /b %errorlevel%
)
echo [Task 3] Running...
task3

echo.
echo ------------------------------------------
echo.

echo [Task 4] Compiling...
nvcc -o task4 task4.cu
if %errorlevel% neq 0 (
    echo Compilation of Task 4 failed!
    exit /b %errorlevel%
)
echo [Task 4] Running...
task4

echo.
echo ==========================================
echo All Tasks Completed.
pause
