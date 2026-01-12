@echo off
echo Building Project...
if not exist build mkdir build
cd build
cmake ..
cmake --build . --config Debug
if %errorlevel% neq 0 (
    echo Build failed!
    exit /b %errorlevel%
)
cd ..

echo.
echo Running Task 1: Vector Add
cd task1_vector_add
..\build\task1_vector_add\Debug\vector_add.exe
cd ..

echo.
echo Running Task 2: Matrix Multiplication
cd task2_matrix_mul
..\build\task2_matrix_mul\Debug\matrix_mul.exe
cd ..
