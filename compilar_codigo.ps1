# --- CONFIGURAÇÕES ---
$arquivoSaida = "relatorio_codigo_fonte.txt"

# 1. Extensões para ignorar
# Adicionei .sql na lista abaixo
$extensoesIgnoradas = @(
    ".db", ".sqlite", ".sqlite3", ".mdb", ".log", ".ldf", ".mdf", 
    ".example", ".md", ".txt", ".pickle", ".bin", 
    ".pyc", ".pyd", ".gguf", ".pt", ".safetensors",
    ".pdf", ".json", ".sql", ".jpg", ".jpeg", ".png"
)

# 2. Pastas para ignorar
$pastasIgnoradas = @(".venv", "models", "ollama_data", "chromadb-data", ".git", "__pycache__", "tests") 

# --- INÍCIO DO SCRIPT ---

# Remove relatório anterior
if (Test-Path $arquivoSaida) { Remove-Item $arquivoSaida }

Write-Host "Iniciando varredura (Ignorando SQL, PDFs, JSONs e binários)..." -ForegroundColor Yellow

# Busca recursiva com filtro duplo
$listaArquivos = Get-ChildItem -Path . -Recurse -File | Where-Object { 
    $arquivo = $_
    
    # A. Verifica extensão
    $extOk = $arquivo.Extension -notin $extensoesIgnoradas
    
    # B. Verifica pastas proibidas no caminho
    $caminhoOk = $true
    foreach ($pasta in $pastasIgnoradas) {
        if ($arquivo.FullName -match "\\$pasta\\") {
            $caminhoOk = $false
            break
        }
    }

    $extOk -and $caminhoOk
}

$total = $listaArquivos.Count
$contador = 0

# Loop de processamento
foreach ($arquivo in $listaArquivos) {
    $contador++
    
    # Progresso visual
    Write-Host "[$contador de $total] Lendo: $($arquivo.Name)" -ForegroundColor Cyan

    # Cabeçalho do arquivo
    Add-Content -Path $arquivoSaida -Value "========================================" -Encoding UTF8
    Add-Content -Path $arquivoSaida -Value "CAMINHO: $($arquivo.FullName)" -Encoding UTF8
    Add-Content -Path $arquivoSaida -Value "========================================" -Encoding UTF8
    
    # Conteúdo
    try {
        Get-Content $arquivo.FullName -Encoding UTF8 -ErrorAction Stop | Add-Content -Path $arquivoSaida -Encoding UTF8
    }
    catch {
        Add-Content -Path $arquivoSaida -Value "[ERRO: Arquivo ilegível ou em uso]" -Encoding UTF8
        Write-Host " -> Erro ao ler: $($arquivo.Name)" -ForegroundColor Red
    }

    # Quebras de linha
    Add-Content -Path $arquivoSaida -Value "`n`n" -Encoding UTF8
}

Write-Host "Processo concluído! Relatório salvo em: $arquivoSaida" -ForegroundColor Green