# -*- coding: utf-8 -*-
"""
benchmark_thesis.py — Automated Thesis Benchmark Suite (MASTER VERSION)
-----------------------------------------------------------------------
Executes:
1. Comparative Study (Local vs SOTA vs FrugalGPT vs Router)
2. Ablation Study (Router vs No-RAG vs No-Rerank vs Random)

Features:
- VRAM Safety (APU Optimized)
- Ground Truth Validation
- Full Statistical Data Logging
"""

import asyncio
import time
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import httpx
import re
import random
from contextlib import contextmanager
from datasets import load_dataset
from tqdm.asyncio import tqdm
from datetime import datetime
from scipy.stats import pointbiserialr

# --- System Imports ---
try:
    from app.app.providers_async import call_model
    from app.app.router_core import route_and_answer
    from app.app.judges import judge_answer
    from app.app.settings_dynamic import settings
except ImportError:
    import sys
    sys.path.append(".")
    from app.app.providers_async import call_model
    from app.app.router_core import route_and_answer
    from app.app.judges import judge_answer
    from app.app.settings_dynamic import settings

# ==============================================================================
# ⚙️ CONFIGURATION
# ==============================================================================
LOCAL_BASELINES = {
    "Local (Gemma 4B)": "ollama/gemma3:4b",
    # "Local (Qwen 8B)": "ollama/qwen3:8b" 
}
MODEL_SOTA = "openai/gpt-5.1"
OLLAMA_API_URL = os.getenv("OLLAMA_HOST", "http://ollama:11434")

PRICE_SOTA_INPUT = 0.005
PRICE_SOTA_OUTPUT = 0.015

SAMPLES_PER_DATASET = 10 
NUM_RUNS = 3
OUTPUT_DIR = "thesis_results"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 🛠️ UTILS FOR ABLATION
# ==============================================================================
@contextmanager
def temporary_setting(key, value):
    """Altera uma configuração temporariamente durante o teste."""
    original_value = settings.get(key)
    try:
        settings.set(key, str(value), actor="benchmark_ablation")
        time.sleep(0.1) # Propagação
        yield
    finally:
        settings.set(key, str(original_value), actor="benchmark_ablation_restore")

# ==============================================================================
# 🧠 OLLAMA MEMORY MANAGER
# ==============================================================================
async def unload_all_ollama_models():
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            ps_response = await client.get(f"{OLLAMA_API_URL}/api/ps")
            if ps_response.status_code == 200:
                running_models = ps_response.json().get('models', [])
                for model in running_models:
                    await client.post(
                        f"{OLLAMA_API_URL}/api/generate", 
                        json={"model": model['name'], "keep_alive": 0}
                    )
            await asyncio.sleep(2)
        except Exception:
            pass

async def force_switch_ollama_model(target_model_name: str):
    clean_target = target_model_name.replace("ollama/", "").split(":")[0]
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            ps_response = await client.get(f"{OLLAMA_API_URL}/api/ps")
            if ps_response.status_code == 200:
                running_models = ps_response.json().get('models', [])
                models_to_kill = [m['name'] for m in running_models if clean_target not in m['name']]
                
                if models_to_kill:
                    for m_name in models_to_kill:
                        await client.post(f"{OLLAMA_API_URL}/api/generate", json={"model": m_name, "keep_alive": 0})
                    
                    for _ in range(10):
                        await asyncio.sleep(1)
                        check = await client.get(f"{OLLAMA_API_URL}/api/ps")
                        current = check.json().get('models', [])
                        if not any(m in [c['name'] for c in current] for m in models_to_kill):
                            break
            
            await asyncio.sleep(3)

            model_loaded = False
            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=600.0) as load_client:
                        resp = await load_client.post(
                            f"{OLLAMA_API_URL}/api/generate", 
                            json={"model": target_model_name.replace("ollama/", ""), "keep_alive": "10m"}
                        )
                        if resp.status_code == 200:
                            model_loaded = True
                            break
                        else:
                            await asyncio.sleep(5)
                except Exception:
                    await asyncio.sleep(5)
            
            if not model_loaded:
                logger.error(f"❌ CRITICAL: Could not load {target_model_name}")

        except Exception as e:
            logger.warning(f"⚠️ Ollama memory switch warning: {repr(e)}")

# ==============================================================================
# 📏 GROUND TRUTH CHECKER
# ==============================================================================
def check_correctness(dataset_name, model_output, reference):
    pred = str(model_output).lower().strip()
    ref = str(reference).lower().strip()
    
    if dataset_name in ["MMLU", "ARC-Challenge", "HellaSwag", "TruthfulQA"]:
        clean_pred = re.sub(r"[\*\(\)]", "", pred)
        matches = re.findall(r"(?:answer|option|choice)?\s*([a-d0-9])\b", clean_pred)
        if matches: return 1 if matches[-1] == ref else 0
        return 0

    if dataset_name == "GSM8K":
        clean_pred = pred.replace(",", "")
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", clean_pred)
        if nums:
            try:
                val = float(nums[-1])
                ref_val = float(ref.replace(",", ""))
                return 1 if abs(val - ref_val) < 1e-6 else 0
            except: pass
        return 0
    
    if dataset_name == "BBH-Date":
        return 1 if ref in pred else 0

    return 0 

# ==============================================================================
# 📚 DATASET LOADERS
# ==============================================================================
def format_mmlu(example):
    options = ["A", "B", "C", "D"]
    choices_str = "\n".join([f"{opt}) {choice}" for opt, choice in zip(options, example['choices'])])
    return f"Question: {example['question']}\nOptions:\n{choices_str}\nAnswer with the correct letter only."

def format_gsm8k(example): return f"Question: {example['question']}\nLet's think step by step."
def format_hellaswag(example):
    options = "\n".join([f"{i+1}) {end}" for i, end in enumerate(example['endings'])])
    return f"Context: {example['ctx']}\nWhich ending makes the most sense?\n{options}\nAnswer with the number only."
def format_humaneval(example): return f"Complete the following Python code:\n\n{example['prompt']}\n    # TODO: implementation"
def format_truthfulqa(example): return f"Question: {example['question']}\nAnswer truthfully and concisely."
def format_arc(example):
    choices = example['choices']['text']
    labels = example['choices']['label']
    choices_str = "\n".join([f"{lbl}) {txt}" for lbl, txt in zip(labels, choices)])
    return f"Question: {example['question']}\nOptions:\n{choices_str}\nAnswer with the correct letter only."
def format_bbh(example): return f"Q: {example['input']}\nA: Let's think step by step."

def load_datasets():
    logger.info("📥 Downloading datasets...")
    tasks = []
    global_id = 1
    datasets_config = [
        ("cais/mmlu", "global_facts", "test", "MMLU", "Knowledge", format_mmlu, lambda x: ["A", "B", "C", "D"][x['answer']]),
        ("gsm8k", "main", "test", "GSM8K", "Reasoning", format_gsm8k, lambda x: x['answer'].split('####')[-1].strip()),
        ("rowan/hellaswag", "default", "validation", "HellaSwag", "Common Sense", format_hellaswag, lambda x: str(int(x['label']) + 1)),
        ("openai_humaneval", None, "test", "HumanEval", "Coding", format_humaneval, lambda x: "CODE_EVAL"),
        ("truthful_qa", "generation", "validation", "TruthfulQA", "Safety", format_truthfulqa, lambda x: x['best_answer']),
        ("ai2_arc", "ARC-Challenge", "test", "ARC-Challenge", "Reasoning", format_arc, lambda x: x['answerKey']),
        ("lukaemon/bbh", "date_understanding", "test", "BBH-Date", "Symbolic Logic", format_bbh, lambda x: x['target'])
    ]
    for path, name, split, label, cat, fmt_func, ref_func in datasets_config:
        try:
            if name: ds = load_dataset(path, name, split=split)
            else: ds = load_dataset(path, split=split)
            ds = ds.shuffle(seed=42).select(range(SAMPLES_PER_DATASET))
            for item in ds:
                tasks.append({"id": global_id, "dataset": label, "category": cat, "query": fmt_func(item), "reference": ref_func(item)})
                global_id += 1
        except Exception as e: logger.error(f"Failed {label}: {e}")
    logger.info(f"✅ Loaded {len(tasks)} tasks.")
    return tasks

# ==============================================================================
# 🌊 FRUGAL GPT
# ==============================================================================
async def run_frugal_cascade(query: str):
    start_t = time.time()
    local_model = list(LOCAL_BASELINES.values())[0]
    await force_switch_ollama_model(local_model)
    
    ans_local, meta_local = await call_model(local_model, query, max_tokens=512, temperature=0.1)
    
    await unload_all_ollama_models()
    judge_res = await judge_answer(query, ans_local)
    scores = [r["score"] for r in judge_res if "score" in r]
    score_local = sum(scores)/len(scores) if scores else 0.0
    
    if score_local >= 8.0:
        return {
            "answer": ans_local, "model": "FrugalGPT (Local)",
            "cost": meta_local.get("cost_per_1k", 0.0),
            "latency": time.time() - start_t,
            "load_time": meta_local.get("load_time", 0.0)
        }
        
    ans_sota, meta_sota = await call_model(MODEL_SOTA, query, max_tokens=512, temperature=0.1)
    total_cost = meta_local.get("cost_per_1k", 0.0) + meta_sota.get("cost_per_1k", 0.0)
    
    return {
        "answer": ans_sota, "model": "FrugalGPT (SOTA)",
        "cost": total_cost, "latency": time.time() - start_t,
        "load_time": meta_local.get("load_time", 0.0)
    }

# ==============================================================================
# 🏃 EXECUTION ENGINE
# ==============================================================================

def estimate_fallback_cost(query: str, answer: str) -> float:
    in_tokens = len(query) / 4
    out_tokens = len(answer) / 4
    return (in_tokens / 1000 * PRICE_SOTA_INPUT) + (out_tokens / 1000 * PRICE_SOTA_OUTPUT)

async def evaluate_interaction(mode_label: str, query: str, reference: str, dataset: str):
    start_t = time.time()
    answer = ""
    model_used = "unknown"
    cost = 0.0
    latency = 0.0
    load_time = 0.0
    
    try:
        # --- 1. ROUTER (FULL) ---
        if mode_label == "Router (Hybrid)":
            res = await route_and_answer(query=query, use_cache=False, use_rag=True)
            answer = res.get("answer", "")
            model_used = res.get("model", "error")
            cost = res.get("cost_per_1k", 0.0)
            load_time = res.get("load_time_s", 0.0)

        # --- 2. FRUGAL GPT ---
        elif mode_label == "FrugalGPT (Cascade)":
            res = await run_frugal_cascade(query)
            answer = res["answer"]
            model_used = res["model"]
            cost = res["cost"]
            total_latency = res["latency"]
            load_time = res["load_time"]
            if "SOTA" in model_used and cost <= 1e-6: cost += estimate_fallback_cost(query, answer)

        # --- 3. ABLATION: NO RAG ---
        elif mode_label == "Router (No RAG)":
            res = await route_and_answer(query=query, use_cache=False, use_rag=False)
            answer = res.get("answer", "")
            model_used = res.get("model", "error")
            cost = res.get("cost_per_1k", 0.0)
            load_time = res.get("load_time_s", 0.0)

        # --- 4. ABLATION: NO RE-RANK ---
        elif mode_label == "Router (No Re-rank)":
            with temporary_setting("RERANK_ENABLED", "0"):
                res = await route_and_answer(query=query, use_cache=False, use_rag=True)
            answer = res.get("answer", "")
            model_used = res.get("model", "error")
            cost = res.get("cost_per_1k", 0.0)
            load_time = res.get("load_time_s", 0.0)

        # --- 5. ABLATION: RANDOM ---
        elif mode_label == "Router (Random)":
            target = random.choice([list(LOCAL_BASELINES.values())[0], MODEL_SOTA])
            if "ollama" in target: await force_switch_ollama_model(target)
            answer, meta = await call_model(target, query, max_tokens=512, temperature=0.1)
            model_used = target
            cost = meta.get("cost_per_1k", 0.0)
            load_time = meta.get("load_time", 0.0)
            if cost <= 1e-6 and "gpt" in target: cost = estimate_fallback_cost(query, answer)

        # --- 6. SOTA ---
        elif mode_label == "SOTA (GPT-5.1)":
            answer, meta = await call_model(MODEL_SOTA, query, max_tokens=512, temperature=0.1)
            model_used = MODEL_SOTA
            cost = meta.get("cost_per_1k", 0.0)
            if cost <= 1e-6: cost = estimate_fallback_cost(query, answer)

        # --- 7. LOCAL ---
        elif mode_label in LOCAL_BASELINES:
            model_id = LOCAL_BASELINES[mode_label]
            await force_switch_ollama_model(model_id)
            answer, meta = await call_model(model_id, query, max_tokens=512, temperature=0.1)
            model_used = model_id
            cost = meta.get("cost_per_1k", 0.0)
            load_time = meta.get("load_time", 0.0)
            
        else:
            raise ValueError(f"Modo desconhecido: {mode_label}")
        
        if mode_label != "FrugalGPT (Cascade)":
            total_latency = time.time() - start_t
        
        effective_latency = max(0.0, total_latency - load_time)

        if "ollama" in model_used or "Local" in model_used:
            await unload_all_ollama_models()

        judge_score = 0.0
        if answer:
            for _ in range(2):
                try:
                    judge_res = await judge_answer(query, answer, reference=reference)
                    scores = [r["score"] for r in judge_res if "score" in r]
                    raw_quality = sum(scores)/len(scores) if scores else 0.0
                    judge_score = raw_quality * 10.0 if raw_quality <= 1.0 else raw_quality
                    break
                except Exception: await asyncio.sleep(2)

        is_correct = check_correctness(dataset, answer, reference)

        return {
            "mode": mode_label, "model_used": model_used,
            "latency": effective_latency, "cost": cost, 
            "judge_score": judge_score, "is_correct": is_correct,
            "success": True,
            "query": query, "answer": answer, "reference": reference
        }

    except Exception as e:
        logger.error(f"Error in {mode_label}: {e}")
        return {
            "mode": mode_label, "model_used": "error",
            "latency": 0, "cost": 0, "judge_score": 0, "is_correct": 0,
            "success": False, "query": query, "answer": "", "reference": reference
        }

async def run_benchmark_suite():
    tasks_data = load_datasets()
    if not tasks_data: return pd.DataFrame()

    results = []
    # LISTA COMPLETA DE MODOS
    modes = list(LOCAL_BASELINES.keys()) + [
        "SOTA (GPT-5.1)", 
        "FrugalGPT (Cascade)", 
        "Router (Hybrid)",
        "Router (No RAG)",
        "Router (No Re-rank)",
        "Router (Random)"
    ]
    
    total_iterations = NUM_RUNS * len(tasks_data) * len(modes)
    pbar = tqdm(total=total_iterations, desc="Thesis Benchmark")

    for run in range(1, NUM_RUNS + 1):
        for task in tasks_data:
            for mode in modes:
                data = await evaluate_interaction(mode, task["query"], task["reference"], task["dataset"])
                data.update({"run_id": run, "id": task["id"], "dataset": task["dataset"], "category": task["category"]})
                results.append(data)
                pbar.update(1)
                await asyncio.sleep(0.5) 

    pbar.close()
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    df.to_csv(f"{OUTPUT_DIR}/raw_data_{timestamp}.csv", index=False)
    return df

# ==============================================================================
# 📊 PLOTTING
# ==============================================================================

def generate_meta_plots(df):
    if df.empty: return
    logger.info("📊 Generating plots...")
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # Paleta Estendida
    palette = {
        "Local (Gemma 4B)": "#95a5a6", # Cinza
        "SOTA (GPT-5.1)": "#3498db",   # Azul
        "FrugalGPT (Cascade)": "#9b59b6", # Roxo
        "Router (Hybrid)": "#2ecc71",    # Verde (Destaque)
        "Router (No RAG)": "#f39c12",  # Laranja
        "Router (No Re-rank)": "#e67e22", # Laranja Escuro
        "Router (Random)": "#e74c3c"   # Vermelho
    }

    # 1. Judge Reliability
    plt.figure(figsize=(8, 6))
    df_valid = df.dropna(subset=['is_correct'])
    df_valid['Ground Truth'] = df_valid['is_correct'].map({1: 'Correct', 0: 'Incorrect'})
    sns.boxplot(data=df_valid, x="Ground Truth", y="judge_score", palette="Set2")
    plt.title("Judge Reliability: Score vs. Ground Truth", fontweight='bold')
    plt.savefig(f"{OUTPUT_DIR}/fig_judge_reliability.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Objective Accuracy (Ablation View)
    plt.figure(figsize=(16, 6))
    sns.barplot(data=df_valid, x="dataset", y="is_correct", hue="mode", palette=palette, errorbar=None)
    plt.title("Ablation Study: Objective Accuracy by Dataset", fontweight='bold')
    plt.ylabel("Accuracy (0.0 - 1.0)")
    plt.ylim(0, 1.0)
    plt.legend(title="Approach", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f"{OUTPUT_DIR}/fig_ablation_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Cost (Log Scale)
    df['plot_cost'] = df['cost'].apply(lambda x: x if x > 0 else 1e-6)
    plt.figure(figsize=(16, 6))
    sns.barplot(data=df, x="dataset", y="plot_cost", hue="mode", palette=palette, errorbar="sd", capsize=.1)
    plt.title("Cost Efficiency (Log Scale)", fontweight='bold')
    plt.ylabel("Cost per Query (USD)")
    plt.yscale("log")
    plt.ylim(bottom=1e-7)
    plt.legend(title="Approach", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f"{OUTPUT_DIR}/fig_cost_logscale.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Latency
    plt.figure(figsize=(16, 6))
    sns.barplot(data=df, x="dataset", y="latency", hue="mode", palette=palette, errorbar="sd", capsize=.1)
    plt.title("Effective Latency per Dataset", fontweight='bold')
    plt.ylabel("Latency (s)")
    plt.legend(title="Approach", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f"{OUTPUT_DIR}/fig_latency.png", dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✅ Plots generated in {OUTPUT_DIR}/")

def calculate_judge_metrics(df):
    df_valid = df.dropna(subset=['is_correct'])
    if df_valid.empty: return
    
    result = pointbiserialr(df_valid['is_correct'], df_valid['judge_score'])
    corr = float(result[0])
    p_val = float(result[1])
    
    print("\n" + "="*60)
    print("⚖️ JUDGE RELIABILITY REPORT")
    print("="*60)
    print(f"Correlation (Point-Biserial): {corr:.4f} (p={p_val:.4e})")
    print("-" * 60)
    print("Average Score when Correct:   ", df_valid[df_valid['is_correct']==1]['judge_score'].mean())
    print("Average Score when Incorrect: ", df_valid[df_valid['is_correct']==0]['judge_score'].mean())
    print("="*60)

if __name__ == "__main__":
    print("🧪 Starting Thesis Benchmark Suite (Full + Ablation)...")
    df_results = asyncio.run(run_benchmark_suite())
    if not df_results.empty:
        generate_meta_plots(df_results)
        calculate_judge_metrics(df_results)
        print(f"\n🏆 Benchmark Complete. Check '{OUTPUT_DIR}' folder.")
    else:
        print("\n❌ Benchmark failed.")