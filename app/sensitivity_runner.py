# -*- coding: utf-8 -*-
"""
sensitivity_runner.py — Sensitivity Analysis for Thesis (ROBUST FIXED)
----------------------------------------------------------------------
Runs the benchmark multiple times with different Uncertainty Thresholds.

FIXES:
- Prevents 'max() arg is an empty sequence' error.
- Prevents Seaborn errors on empty DataFrames.
- Ensures clean state between runs.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import time
import glob
import logging
import shutil

# Configuração
THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]
OUTPUT_FILE = "thesis_results/sensitivity_data.csv"
CHECKPOINT_PATH = "thesis_results/benchmark_checkpoint.csv"
RAW_DATA_PATTERN = "thesis_results/raw_data_*.csv"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("sensitivity")

def run_benchmark_iteration(threshold):
    """Executa benchmark iteration."""
    logger.info(f"\n⚙️  Testing UNCERTAINTY_THRESHOLD = {threshold}...")
    
    # 1. Limpeza Crítica: Remove checkpoint anterior
    if os.path.exists(CHECKPOINT_PATH):
        try:
            os.remove(CHECKPOINT_PATH)
        except Exception as e:
            logger.warning(f"Could not remove old checkpoint: {e}")

    # 2. Prepara Ambiente
    env = os.environ.copy()
    env["UNCERTAINTY_THRESHOLD"] = str(threshold)
    env["NUM_RUNS"] = "2" 
    env["SAMPLES_PER_DATASET"] = "10" 
    
    # 3. Executa o Benchmark
    try:
        # Capture output=True esconde o log do benchmark para não poluir, 
        # mude para False se quiser ver o progresso detalhado
        subprocess.run(["python", "benchmark_thesis.py"], env=env, check=True, capture_output=False)
    except subprocess.CalledProcessError as e:
        logger.error(f"Benchmark crashed for threshold {threshold}: {e}")
        return None
    
    # 4. Carrega Resultados (Com verificação de segurança)
    df = None
    
    # Tenta achar o arquivo final timestamped
    files = glob.glob(RAW_DATA_PATTERN)
    
    if files:
        # Pega o mais recente
        latest = max(files, key=os.path.getctime)
        # Verifica se foi criado nos últimos 2 minutos (para não pegar arquivo de run anterior)
        if time.time() - os.path.getctime(latest) < 300:
            logger.info(f"📂 Reading from raw data: {latest}")
            df = pd.read_csv(latest)
    
    # Fallback para o checkpoint se o raw_data não for recente ou não existir
    if df is None:
        if os.path.exists(CHECKPOINT_PATH):
            logger.info(f"⚠️ Raw data missing/old. Falling back to checkpoint: {CHECKPOINT_PATH}")
            df = pd.read_csv(CHECKPOINT_PATH)
        else:
            logger.error(f"❌ No data found for threshold {threshold}")
            return None

    # 5. Normalização de Colunas (Garante compatibilidade)
    if 'judge_score' in df.columns and 'quality' not in df.columns:
        df.rename(columns={'judge_score': 'quality'}, inplace=True)
    
    if 'quality' not in df.columns:
        logger.error(f"❌ Column 'quality' missing in data for threshold {threshold}")
        return None

    # 6. Filtra dados do Router
    # Tenta encontrar qualquer modo que seja o Router
    router_df = df[df['mode'].str.contains("Router", case=False, na=False)]
    
    if router_df.empty:
        logger.warning(f"⚠️ No Router data found for threshold {threshold}")
        return None

    return {
        "threshold": threshold,
        "avg_cost": router_df['cost'].mean(),
        "avg_quality": router_df['quality'].mean(),
        "avg_latency": router_df['latency'].mean()
    }

def plot_sensitivity(results):
    """Executa plot sensitivity."""
    if not results:
        logger.error("❌ No valid results to plot.")
        return

    df = pd.DataFrame(results)
    
    # Verificação final de integridade do DataFrame
    if 'threshold' not in df.columns or df.empty:
        logger.error("❌ DataFrame is empty or missing 'threshold' column.")
        return

    df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"💾 Sensitivity data saved to {OUTPUT_FILE}")
    
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Eixo Y1: Custo
    sns.lineplot(data=df, x="threshold", y="avg_cost", ax=ax1, color="#e74c3c", marker="o", label="Cost ($)", linewidth=2.5)
    ax1.set_ylabel("Average Cost ($)", color="#e74c3c", fontweight='bold')
    ax1.tick_params(axis='y', labelcolor="#e74c3c")
    ax1.set_xlabel("Uncertainty Threshold (tau)", fontweight='bold')
    
    # Eixo Y2: Qualidade
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x="threshold", y="avg_quality", ax=ax2, color="#2ecc71", marker="s", label="Quality (0-10)", linewidth=2.5)
    ax2.set_ylabel("Average Quality (0-10)", color="#2ecc71", fontweight='bold')
    ax2.tick_params(axis='y', labelcolor="#2ecc71")
    
    plt.title("Sensitivity Analysis: Impact of Uncertainty Threshold", fontweight='bold', fontsize=14)
    
    # Legendas manuais para evitar duplicação
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    # Filtra legendas vazias
    if lines_1 and lines_2:
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

    plt.grid(True, alpha=0.3)
    
    output_img = "thesis_results/fig_sensitivity_analysis.png"
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Sensitivity plot saved: {output_img}")

if __name__ == "__main__":
    logger.info("🧪 Starting Sensitivity Analysis Runner...")
    
    aggregated_results = []
    
    for t in THRESHOLDS:
        try:
            res = run_benchmark_iteration(t)
            if res:
                aggregated_results.append(res)
            else:
                logger.warning(f"⚠️ Run for threshold {t} returned no data.")
        except Exception as e:
            logger.error(f"❌ Critical failure for threshold {t}: {e}")
            
    if aggregated_results:
        plot_sensitivity(aggregated_results)
    else:
        logger.error("❌ All runs failed. No graph generated.")