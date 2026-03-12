#!/bin/bash
# Objective: Shell utility for consolidar.

# Nome do arquivo de saída
OUTPUT="consolidado.txt"

# Limpa o arquivo de saída caso ele já exista
> "$OUTPUT"

# Busca arquivos .py e .sql
# O comando find busca recursivamente a partir do diretório atual (.)
find . -type f \( -name "*.py" \) | while read -r FILE; do
    
    # Obtém o caminho completo (absoluto) do arquivo
    FULL_PATH=$(realpath "$FILE")
    
    # 1. Insere o caminho completo e o nome
    echo "ARQUIVO: $FULL_PATH" >> "$OUTPUT"
    
    # 2. Insere duas linhas em branco
    echo -e "\n" >> "$OUTPUT"
    
    # 3. Insere o conteúdo do arquivo
    cat "$FILE" >> "$OUTPUT"
    
    # Adiciona uma quebra de linha extra ao final para separar do próximo arquivo
    echo -e "\n\n--- FIM DO ARQUIVO ---\n" >> "$OUTPUT"

done

echo "Concluído! O conteúdo foi salvo em: $OUTPUT"
