# -*- coding: utf-8 -*-
"""
benchmark_thesis.py — Automated Thesis Benchmark Suite
------------------------------------------------------
Executes a comparative study between Local, SOTA, and Hybrid Router approaches.
Datasets: MMLU, GSM8K, HellaSwag.
Repetitions: 5 runs per dataset to capture stochastic variance.
Output: CSV raw data + High-Quality Plots with Standard Deviation.
"""

import asyncio
import time
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from tqdm.asyncio import tqdm
from datetime import datetime

# --- System Imports ---
try:
    from app.providers_async import call_model
    from app.router_core import route_and_answer
    from app.judges import judge_answer
except ImportError:
    import sys
    sys.path.append(".")
    from app.providers_async import call_model
    from app.router_core import route_and_answer
    from app.judges import judge_answer

# ==============================================================================
# ⚙️ CONFIGURATION
# ==============================================================================
# Ajuste os nomes dos modelos conforme seu .env
MODEL_LOCAL = "ollama/gemma3:4b" 
MODEL_SOTA = "openai/gpt-5.1"

# Configuração da Execução
SAMPLES_PER_DATASET = 10  # Aumente para 30-50 na versão final
NUM_RUNS = 3
OUTPUT_DIR = "thesis_results"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 📚 DATASET LOADERS & FORMATTERS
# ==============================================================================

def format_mmlu(example):
    options = ["A", "B", "C", "D"]
    choices_str = "\n".join([f"{opt}) {choice}" for opt, choice in zip(options, example['choices'])])
    return f"Question: {example['question']}\nOptions:\n{choices_str}\nAnswer with the correct letter only."

def format_gsm8k(example):
    return f"Question: {example['question']}\nLet's think step by step."

def format_hellaswag(example):
    options = "\n".join([f"{i+1}) {end}" for i, end in enumerate(example['endings'])])
    return f"Context: {example['ctx']}\nWhich ending makes the most sense?\n{options}\nAnswer with the number only."

def load_datasets():
    logger.info("📥 Downloading and preparing datasets (Hugging Face)...")
    tasks = []
    
    # 1. MMLU (Knowledge)
    try:
        # Removido trust_remote_code=True para evitar erro em versões novas
        ds_mmlu = load_dataset("cais/mmlu", "global_facts", split="test")
        ds_mmlu = ds_mmlu.shuffle(seed=42).select(range(SAMPLES_PER_DATASET))
        for item in ds_mmlu:
            tasks.append({
                "dataset": "MMLU",
                "category": "Knowledge",
                "query": format_mmlu(item),
                "reference": ["A", "B", "C", "D"][item['answer']]
            })
    except Exception as e:
        logger.error(f"Failed to load MMLU: {e}")

    # 2. GSM8K (Reasoning)
    try:
        ds_gsm = load_dataset("gsm8k", "main", split="test")
        ds_gsm = ds_gsm.shuffle(seed=42).select(range(SAMPLES_PER_DATASET))
        for item in ds_gsm:
            tasks.append({
                "dataset": "GSM8K",
                "category": "Reasoning",
                "query": format_gsm8k(item),
                "reference": item['answer']
            })
    except Exception as e:
        logger.error(f"Failed to load GSM8K: {e}")

    # 3. HellaSwag (Common Sense)
    try:
        ds_hella = load_dataset("rowan/hellaswag", split="validation")
        ds_hella = ds_hella.shuffle(seed=42).select(range(SAMPLES_PER_DATASET))
        for item in ds_hella:
            tasks.append({
                "dataset": "HellaSwag",
                "category": "Common Sense",
                "query": format_hellaswag(item),
                "reference": item['label']
            })
    except Exception as e:
        logger.error(f"Failed to load HellaSwag: {e}")

    logger.info(f"✅ Loaded {len(tasks)} total tasks.")
    return tasks

# ==============================================================================
# 🏃 EXECUTION ENGINE
# ==============================================================================

async def evaluate_interaction(mode: str, query: str, run_id: int):
    start_t = time.time()
    
    # Inicializa variáveis para evitar UnboundLocalError
    answer = ""
    model = "unknown"
    cost = 0.0
    latency = 0.0
    
    try:
        if mode == "Router (Hybrid)":
            res = await route_and_answer(query=query, use_cache=False)
            answer = res.get("answer", "")
            model = res.get("model", "router_error")
            cost = res.get("cost_per_1k", 0.0)
        
        elif mode == "Local (Gemma 3)":
            answer, meta = await call_model(MODEL_LOCAL, query, max_tokens=512, temperature=0.1)
            model = MODEL_LOCAL
            cost = meta.get("cost_per_1k", 0.0)
        
        elif mode == "SOTA (GPT-5.1)":
            answer, meta = await call_model(MODEL_SOTA, query, max_tokens=512, temperature=0.1)
            model = MODEL_SOTA
            cost = meta.get("cost_per_1k", 0.0)
            
        else:
            raise ValueError(f"Modo desconhecido: {mode}")
        
        latency = time.time() - start_t

        # Auto-Evaluation (LLM-as-a-Judge)
        if answer:
            judge_res = await judge_answer(query, answer)
            scores = [r["score"] for r in judge_res if "score" in r]
            quality = sum(scores)/len(scores) if scores else 0.0
        else:
            quality = 0.0

        return {
            "run_id": run_id,
            "mode": mode,
            "model_used": model,
            "latency": latency,
            "cost": cost,
            "quality": quality,
            "success": True
        }

    except Exception as e:
        logger.error(f"Error in {mode}: {e}")
        return {
            "run_id": run_id,
            "mode": mode,
            "model_used": "error",
            "latency": 0,
            "cost": 0,
            "quality": 0,
            "success": False,
            "error_msg": str(e)
        }

async def run_benchmark_suite():
    tasks_data = load_datasets()
    if not tasks_data:
        logger.error("No datasets loaded. Exiting.")
        return pd.DataFrame()

    results = []
    # Estes nomes devem bater EXATAMENTE com os ifs em evaluate_interaction
    modes = ["Local (Gemma 3)", "SOTA (GPT-5.1)", "Router (Hybrid)"]
    
    total_iterations = NUM_RUNS * len(tasks_data) * len(modes)
    pbar = tqdm(total=total_iterations, desc="Thesis Benchmark Progress")

    for run in range(1, NUM_RUNS + 1):
        for task in tasks_data:
            for mode in modes:
                data = await evaluate_interaction(mode, task["query"], run)
                data.update({
                    "dataset": task["dataset"],
                    "category": task["category"]
                })
                results.append(data)
                pbar.update(1)
                # Breve pausa para evitar rate limit agressivo
                await asyncio.sleep(0.05) 

    pbar.close()
    
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = f"{OUTPUT_DIR}/raw_data_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"💾 Raw data saved to {csv_path}")
    
    return df

# ==============================================================================
# 📊 PLOTTING (With Standard Deviation)
# ==============================================================================

def generate_thesis_plots(df):
    if df.empty:
        logger.warning("DataFrame is empty. Skipping plots.")
        return

    logger.info("📊 Generating plots with Standard Deviation...")
    
    # Configuração de Estilo
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    palette = {"Local (Gemma 3)": "#e74c3c", "SOTA (GPT-5.1)": "#3498db", "Router (Hybrid)": "#2ecc71"}
    
    # --- PLOT 1: Pareto Frontier (Mean + SD Error Bars) ---
    plt.figure(figsize=(10, 7))
    
    # Agrupa dados globais
    df_global = df.groupby("mode").agg({
        "quality": ["mean", "std"],
        "cost": ["mean", "std"]
    }).reset_index()
    
    # Flatten columns
    df_global.columns = ["mode", "qual_mean", "qual_std", "cost_mean", "cost_std"]

    # Plot manual com errorbar para ter controle total X e Y
    for i, row in df_global.iterrows():
        color = palette.get(row['mode'], "gray")
        plt.errorbar(
            x=row['cost_mean'], 
            y=row['qual_mean'], 
            xerr=row['cost_std'], 
            yerr=row['qual_std'], 
            fmt='o', 
            color=color, 
            ecolor='gray', 
            elinewidth=2, 
            capsize=5, 
            markersize=15,
            label=row['mode']
        )
        # Anotação
        plt.text(
            row['cost_mean'], 
            row['qual_mean'] + 0.15, 
            row['mode'], 
            ha='center', 
            fontdict={'size': 12, 'weight': 'bold'}
        )

    plt.title("Cost-Quality Pareto Frontier (Mean ± SD)", fontweight='bold')
    plt.xlabel("Average Cost per Query (USD)")
    plt.ylabel("Average Quality Score (0-10)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Approach", loc='lower right')
    
    # Ajuste de margem para caber as barras de erro
    plt.margins(0.1)
    plt.savefig(f"{OUTPUT_DIR}/fig1_pareto_frontier_with_sd.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- PLOT 2: Quality by Dataset (Barplot with SD) ---
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df, 
        x="dataset", 
        y="quality", 
        hue="mode", 
        palette=palette, 
        errorbar="sd",  # Standard Deviation
        capsize=.1
    )
    plt.title("Quality Performance across Datasets (Mean ± SD)", fontweight='bold')
    plt.ylabel("Quality Score (Judge 0-10)")
    plt.xlabel("Benchmark Dataset")
    plt.legend(title="Approach", loc='lower right')
    plt.ylim(0, 11) # Espaço para a barra de erro superior
    plt.savefig(f"{OUTPUT_DIR}/fig2_quality_by_dataset_sd.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- PLOT 3: Latency Comparison (Barplot with SD) ---
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df, 
        x="mode", 
        y="latency", 
        palette=palette,
        errorbar="sd",
        capsize=.1
    )
    plt.title("Latency Analysis (Mean ± SD)", fontweight='bold')
    plt.ylabel("Latency (seconds)")
    plt.xlabel("Approach")
    plt.savefig(f"{OUTPUT_DIR}/fig3_latency_barplot_sd.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- PLOT 4: Cost Efficiency (Barplot with SD) ---
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df, 
        x="dataset", 
        y="cost", 
        hue="mode", 
        palette=palette,
        errorbar="sd",
        capsize=.1
    )
    plt.title("Cost Efficiency per Dataset (Mean ± SD)", fontweight='bold')
    plt.ylabel("Cost per Query (USD)")
    plt.yscale("log") # Log scale é essencial aqui
    plt.savefig(f"{OUTPUT_DIR}/fig4_cost_logscale_sd.png", dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✅ All plots generated in {OUTPUT_DIR}/")

# ==============================================================================
# 📋 SUMMARY TABLE GENERATOR
# ==============================================================================

def print_statistical_summary(df):
    if df.empty: return

    print("\n" + "="*80)
    print("🏆 STATISTICAL SUMMARY (Mean ± Std Dev)")
    print("="*80)

    # Agrupa e calcula Mean e Std
    summary = df.groupby("run_type").agg({
        "quality": ["mean", "std"],
        "latency": ["mean", "std"],
        "cost": ["mean", "std"]
    })

    # Formata para string legível "Média ± Desvio"
    final_table = pd.DataFrame()
    final_table["Quality (0-10)"] = summary["quality"].apply(lambda x: f"{x['mean']:.2f} ± {x['std']:.2f}", axis=1)
    final_table["Latency (s)"] = summary["latency"].apply(lambda x: f"{x['mean']:.2f} ± {x['std']:.2f}", axis=1)
    final_table["Cost ($)"] = summary["cost"].apply(lambda x: f"{x['mean']:.5f} ± {x['std']:.5f}", axis=1)

    print(final_table.to_string())
    print("-" * 80)
    
    # Cálculo de Economia (usando as médias)
    try:
        mean_cost_sota = summary.loc["sota", ("cost", "mean")]
        mean_cost_router = summary.loc["router", ("cost", "mean")]
        if mean_cost_sota > 0:
            savings = 100 * (1 - (mean_cost_router / mean_cost_sota))
            print(f"💰 Average Cost Savings (Router vs SOTA): {savings:.2f}%")
        else:
            print("💰 Cost Savings: N/A (SOTA cost is 0)")
    except KeyError:
        print("⚠️ Could not calculate savings (missing data for sota or router)")
        
    print("="*80)

# ==============================================================================
# 🚀 MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    print("🧪 Starting Thesis Benchmark Suite (Stochastic Analysis)...")
    print(f"   - Runs: {NUM_RUNS}")
    print(f"   - Samples/Dataset: {SAMPLES_PER_DATASET}")
    
    # 1. Run Benchmark
    df_results = asyncio.run(run_benchmark_suite())
    
    if not df_results.empty:
        # 2. Mapeia nomes internos para nomes de exibição
        df_results["run_type"] = df_results["mode"].map({
            "Local (Gemma 3)": "local",
            "SOTA (GPT-5.1)": "sota",
            "Router (Hybrid)": "router"
        })

        # 3. Generate Plots
        generate_thesis_plots(df_results)
        
        # 4. Print Summary
        print_statistical_summary(df_results)
        
        print(f"\n🏆 Benchmark Complete. Check '{OUTPUT_DIR}' folder.")
    else:
        print("\n❌ Benchmark failed to produce results.")