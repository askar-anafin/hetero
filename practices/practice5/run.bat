@echo off
set "VSCMD_START_DIR=%CD%"
echo Setting up Visual Studio Environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul
cd /d "%~dp0"

echo Building Practice 5 (Basic Stack/Queue)...
nvcc stack_queue.cu -o stack_queue.exe
if %errorlevel% neq 0 (
    echo Build failed for stack_queue.cu!
    pause
    exit /b %errorlevel%
)
echo Build successful! Running stack_queue.exe...
.\stack_queue.exe

echo.
echo Building Practice 5 (Optimized Stack/Queue)...
nvcc stack_queue_opt.cu -o stack_queue_opt.exe
if %errorlevel% neq 0 (
    echo Build failed for stack_queue_opt.cu!
    pause
    exit /b %errorlevel%
)
echo Build successful! Running stack_queue_opt.exe...
.\stack_queue_opt.exe

echo.
echo Running Visualization...
python plot_results.py

pause
