#!/bin/bash

OUTPUT_FILE="monitoramento_hardware.csv"
INTERVALO=10

# Força o ponto como separador decimal para não quebrar o CSV
export LC_NUMERIC=C

if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Data,Hora,CPU_Temp_C,CPU_Clock_MHz,RTX5070Ti_Temp_C,RX580_Temp_C" > "$OUTPUT_FILE"
fi

echo "Monitorando... Gravando em $OUTPUT_FILE"
echo "Pressione [CTRL+C] para parar."

while true; do
    DATA=$(date '+%Y-%m-%d')
    HORA=$(date '+%H:%M:%S')

    # 1. Temperatura da CPU (Especificamente o Tctl do driver k10temp)
    # Buscamos a linha após 'k10temp' que contenha 'Tctl'
    CPU_TEMP=$(sensors | grep -A 5 "k10temp" | grep "Tctl" | awk -F: '{print $2}' | grep -oE '[0-9]+(\.[0-9]+)?' | head -n 1)
    
    # 2. Clock do Processador (Média de todos os núcleos em MHz)
    CPU_CLOCK=$(awk '{sum += $1} END {printf "%.2f", sum/NR/1000}' /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null)
    
    # 3. Temperatura GPU NVIDIA (RTX 5070 Ti)
    GPU_NV_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -n 1 | xargs)

    # 4. Temperatura GPU AMD (RX 580 - sensor 'edge')
    GPU_AMD_TEMP=$(sensors | grep -A 10 "amdgpu" | grep "edge" | awk -F: '{print $2}' | grep -oE '[0-9]+(\.[0-9]+)?' | head -n 1)

    # Fallback para evitar campos vazios no CSV
    [ -z "$CPU_TEMP" ] && CPU_TEMP="0.0"
    [ -z "$CPU_CLOCK" ] && CPU_CLOCK="0.0"
    [ -z "$GPU_NV_TEMP" ] && GPU_NV_TEMP="0.0"
    [ -z "$GPU_AMD_TEMP" ] && GPU_AMD_TEMP="0.0"

    # Salva no CSV
    echo "$DATA,$HORA,$CPU_TEMP,$CPU_CLOCK,$GPU_NV_TEMP,$GPU_AMD_TEMP" >> "$OUTPUT_FILE"

    # Exibe no terminal para conferência
    echo "------------------------------------------"
    echo "Hora: $HORA"
    echo "CPU Temp: $CPU_TEMP °C | Clock: $CPU_CLOCK MHz"
    echo "RTX 5070 Ti: $GPU_NV_TEMP °C"
    echo "RX 580: $GPU_AMD_TEMP °C"

    sleep $INTERVALO
done
