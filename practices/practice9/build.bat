@echo off
REM Build script for Practice 9 (MPI)
REM Assumes MS-MPI is installed.

echo Setting up Visual Studio environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo.
echo ==========================================
echo Compiling Task 1: Stats
echo ==========================================
cl /EHsc /O2 task1_stats.cpp /I "C:\Program Files (x86)\Microsoft SDKs\MPI\Include" /link /LIBPATH:"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" msmpi.lib
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ==========================================
echo Compiling Task 2: Gaussian Elimination
echo ==========================================
cl /EHsc /O2 task2_gauss.cpp /I "C:\Program Files (x86)\Microsoft SDKs\MPI\Include" /link /LIBPATH:"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" msmpi.lib
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ==========================================
echo Compiling Task 3: Floyd-Warshall
echo ==========================================
cl /EHsc /O2 task3_floyd.cpp /I "C:\Program Files (x86)\Microsoft SDKs\MPI\Include" /link /LIBPATH:"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" msmpi.lib
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ==========================================
echo Build Successful!
echo ==========================================
echo.
echo Run examples:
echo mpiexec -n 4 task1_stats.exe
echo mpiexec -n 4 task2_gauss.exe 8
echo mpiexec -n 4 task3_floyd.exe 4
