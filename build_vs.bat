@echo off
echo Setting up Visual Studio 2022 Environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

if not exist data mkdir data

echo ========================================================
echo Compiling High Precision Proton Therapy MC Engine (2D)...
echo ========================================================

cl.exe /EHsc /O2 /openmp /std:c++17 /I src\mc_engine /Fe:mc_engine.exe ^
    src\mc_engine\main.cpp ^
    src\mc_engine\geometry.cpp ^
    src\mc_engine\physics.cpp ^
    src\mc_engine\transport.cpp ^
    src\mc_engine\export.cpp ^
    src\mc_engine\density_map.cpp

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Compilation Failed!
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] Build complete! Executable: mc_engine.exe
echo   Usage 1D: mc_engine.exe [energy] [shift] [output_dir]
echo   Usage 2D: mc_engine.exe [energy] 0.0 [output_dir] --density-map [phantom.bin]
