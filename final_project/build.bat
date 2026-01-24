@echo off
echo Building Final Project...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
nvcc -o final_project.exe main.cu -Xcompiler "/openmp:llvm /O2"
if %errorlevel% neq 0 (
    echo Build failed!
    exit /b %errorlevel%
)
echo Build successful.
echo Running Final Project...
final_project.exe
echo Generating Plots...
python plot_results.py
