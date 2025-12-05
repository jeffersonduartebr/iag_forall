# -*- coding: utf-8 -*-
"""
sensitivity_runner.py — Sensitivity Analysis for Thesis
-------------------------------------------------------
Runs the benchmark multiple times with different Uncertainty Thresholds
to prove system stability.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import time

# Configuração
THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]
OUTPUT_FILE = "thesis_results/sensitivity_data.csv"

def set_threshold(val):
    # Usa a API de Admin para setar o valor dinamicamente
    # Ou altera via variável de ambiente se o benchmark ler de lá
    os.environ["UNCERTAINTY_THRESHOLD"] = str(val)
    print(f"⚙️  Set UNCERTAINTY_THRESHOLD = {val}")

def run_benchmark_iteration(threshold):
    print(f"\n🚀 Running Benchmark for Threshold {threshold}...")
    
    # Chama o script principal como subprocesso para garantir limpeza
    # Passa a env var para ele pegar
    env = os.environ.copy()
    env["UNCERTAINTY_THRESHOLD"] = str(threshold)
    env["NUM_RUNS"] = "1" # 1 run por threshold é suficiente para sensibilidade
    
    subprocess.run(["python", "benchmark_thesis.py"], env=env, check=True)
    
    # Pega o último CSV gerado
    import glob
    files = glob.glob("thesis_results/raw_data_*.csv")
    latest = max(files, key=os.path.getctime)
    
    df = pd.read_csv(latest)
    router_df = df[df['mode'] == 'Router (Hybrid)']
    
    return {
        "threshold": threshold,
        "avg_cost": router_df['cost'].mean(),
        "avg_quality": router_df['quality'].mean(),
        "avg_latency": router_df['latency'].mean()
    }

def plot_sensitivity(results):
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    
    fig, ax1 = plt.figure(figsize=(10, 6)), plt.gca()
    
    # Eixo Y1: Custo
    sns.lineplot(data=df, x="threshold", y="avg_cost", ax=ax1, color="red", marker="o", label="Cost")
    ax1.set_ylabel("Average Cost ($)", color="red")
    ax1.tick_params(axis='y', labelcolor="red")
    
    # Eixo Y2: Qualidade
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x="threshold", y="avg_quality", ax=ax2, color="green", marker="s", label="Quality")
    ax2.set_ylabel("Average Quality (0-10)", color="green")
    ax2.tick_params(axis='y', labelcolor="green")
    
    plt.title("Sensitivity Analysis: Uncertainty Threshold Impact")
    ax1.set_xlabel("Uncertainty Threshold (tau)")
    plt.grid(True, alpha=0.3)
    
    plt.savefig("thesis_results/fig_sensitivity_analysis.png", dpi=300)
    print("✅ Sensitivity plot saved.")

if __name__ == "__main__":
    results = []
    for t in THRESHOLDS:
        try:
            res = run_benchmark_iteration(t)
            results.append(res)
        except Exception as e:
            print(f"❌ Failed run for {t}: {e}")
            
    plot_sensitivity(results)