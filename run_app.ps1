# Atajo PowerShell para Windows: levanta la app sin necesidad de make.
# Uso desde PowerShell: .\run_app.ps1
param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"

Write-Host "Sincronizando dependencias con uv..." -ForegroundColor Cyan
uv sync

Write-Host "Levantando Streamlit en http://localhost:$Port" -ForegroundColor Green
uv run streamlit run app/Home.py --server.port $Port --server.address 0.0.0.0
