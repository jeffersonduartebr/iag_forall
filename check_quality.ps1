# check_quality.ps1
Write-Host "🔍 Iniciando Verificação de Qualidade de Código..." -ForegroundColor Cyan

# 1. Ruff (Linting)
Write-Host "`n[1/3] Executando Ruff (Linter)..." -ForegroundColor Yellow
ruff check . --fix
if ($LASTEXITCODE -eq 0) { Write-Host "✅ Ruff passou!" -ForegroundColor Green }
else { Write-Host "❌ Ruff encontrou erros." -ForegroundColor Red }

# 2. Ruff (Formatting)
Write-Host "`n[2/3] Executando Ruff (Formatter)..." -ForegroundColor Yellow
ruff format .
if ($LASTEXITCODE -eq 0) { Write-Host "✅ Formatação concluída!" -ForegroundColor Green }

# 3. MyPy (Type Checking)
Write-Host "`n[3/3] Executando MyPy (Type Checker)..." -ForegroundColor Yellow
mypy app/
if ($LASTEXITCODE -eq 0) { Write-Host "✅ MyPy passou!" -ForegroundColor Green }
else { Write-Host "⚠️ MyPy encontrou problemas de tipagem." -ForegroundColor Magenta }

Write-Host "`n🏁 Verificação concluída." -ForegroundColor Cyan