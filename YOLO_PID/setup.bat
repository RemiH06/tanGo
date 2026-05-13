@echo off
echo Creando venv...
python -m venv venv

echo Activando venv...
call venv\Scripts\activate.bat

echo Instalando dependencias...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Listo. Para activar el venv en el futuro:
echo   call venv\Scripts\activate.bat
echo Para correr:
echo   python semaforo_detector.py
pause
