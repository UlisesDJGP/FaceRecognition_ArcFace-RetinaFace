@echo off
echo ============================================================
echo Compilando Motor C++ ONNX Bridge (srf_onnx.dll)
echo ============================================================
echo.

:: Verificar que CMake esté instalado
where cmake >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] CMake no esta instalado o no esta en el PATH.
    echo Descarga CMake desde: https://cmake.org/download/
    pause
    exit /b
)

:: Verificar que el venv tenga onnxruntime
if not exist "venv\Lib\site-packages\onnxruntime\capi\onnxruntime.dll" (
    echo [ERROR] No se encontro onnxruntime en el venv.
    echo Ejecuta: venv\Scripts\pip install onnxruntime-gpu
    pause
    exit /b
)

:: Crear carpeta de build
if not exist "build" mkdir build

:: Configurar con CMake (usa Visual Studio si esta disponible, sino Ninja)
echo Configurando CMake...
cmake -S . -B build

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo la configuracion de CMake.
    echo Asegurate de tener Visual Studio C++ Build Tools instalado.
    pause
    exit /b
)

:: Compilar
echo Compilando...
cmake --build build --config Release

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo la compilacion.
    pause
    exit /b
)

echo.
echo ============================================================
echo Compilacion exitosa! La DLL esta en build\Release\srf_onnx.dll
echo ============================================================
pause
