@echo off
REM Atajo CMD para Windows: levanta la app sin necesidad de make.
REM Uso: run_app.bat
echo Sincronizando dependencias con uv...
uv sync
if errorlevel 1 exit /b %errorlevel%

echo Levantando Streamlit en http://localhost:8501
uv run streamlit run app/Home.py --server.port 8501 --server.address 0.0.0.0
