@echo off
set "VSCMD_START_DIR=%CD%"
echo Setting up Visual Studio Environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul

echo Compiling with NVCC...
nvcc -x cu -Xcompiler "/openmp" main.cpp -o main.exe

if %errorlevel% neq 0 (
    echo Compilation Failed!
    exit /b %errorlevel%
)

echo Running...
echo ----------------------------------------------------------------
main.exe
