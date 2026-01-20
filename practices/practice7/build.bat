@echo off
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
nvcc -o practice7 main.cu
if %errorlevel% neq 0 (
    echo Build failed.
    exit /b %errorlevel%
)
echo Build success.
practice7.exe
