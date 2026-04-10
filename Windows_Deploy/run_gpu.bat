@echo off
echo ========================================================
echo Iniciando FaceRecognition en modo GPU (NVIDIA) en Windows
echo ========================================================

:: Verificamos si existe el entorno virtual
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] No se encontro el entorno virtual en "venv".
    echo Por favor revisa el archivo Instrucciones_Instalacion_Windows.md
    pause
    exit /b
)

:: Habilitamos el tag de GPU
set USE_GPU=1

:: Arrancamos el script principal usando el interprete del entorno local
echo 🔧 Arrancando Inferencia...
"venv\Scripts\python.exe" main.py

pause
