@echo off
set "VSCMD_START_DIR=%CD%"
echo Setting up Visual Studio Environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul
cd /d "%~dp0"

echo Building Practice 3...
nvcc main.cu -o practice3.exe -rdc=true -lcudadevrt -arch=sm_75
if %errorlevel% neq 0 (
    echo Build failed!
    exit /b %errorlevel%
)
echo Build successful! Running...
practice3.exe
pause
