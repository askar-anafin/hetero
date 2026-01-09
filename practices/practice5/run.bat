@echo off & :: Turn off command echoing to keep output clean
set "VSCMD_START_DIR=%CD%" & :: Save the current directory to VSCMD_START_DIR variable
echo Setting up Visual Studio Environment... & :: Print message about VS setup
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul & :: Call the VS environment setup script for 64-bit build, suppressing output
cd /d "%~dp0" & :: Change directory to the script's location

echo Building Practice 5 (Basic Stack/Queue)... & :: Print message about building the basic code
nvcc stack_queue.cu -o stack_queue.exe & :: Compile stack_queue.cu using NVCC
if %errorlevel% neq 0 ( & :: Check if the previous command failed (exit code not 0)
    echo Build failed for stack_queue.cu! & :: Print failure message
    pause & :: Pause to let user see the error
    exit /b %errorlevel% & :: Exit the script with the error code
)
echo Build successful! Running stack_queue.exe... & :: Print success message
.\stack_queue.exe & :: Run the compiled executable

echo. & :: Print an empty line for spacing
echo Building Practice 5 (Optimized Stack/Queue)... & :: Print message about building the optimized code
nvcc stack_queue_opt.cu -o stack_queue_opt.exe & :: Compile stack_queue_opt.cu using NVCC
if %errorlevel% neq 0 ( & :: Check if the previous command failed
    echo Build failed for stack_queue_opt.cu! & :: Print failure message
    pause & :: Pause to let user see the error
    exit /b %errorlevel% & :: Exit the script with the error code
)
echo Build successful! Running stack_queue_opt.exe... & :: Print success message
.\stack_queue_opt.exe & :: Run the compiled executable

pause & :: Pause at the end to keep window open
