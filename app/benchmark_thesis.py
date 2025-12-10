# -*- coding: utf-8 -*-
"""
benchmark_thesis.py — Automated Thesis Benchmark Suite (DYNAMIC PARALLELISM)
----------------------------------------------------------------------------
Executes comparative study with optimized concurrency based on model size.

UPDATES:
- Dynamic Semaphore: 
  - Light Models (Gemma/Granite) -> Run 4x parallel.
  - Heavy Models (Phi-4) -> Run 1x serial.
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
    from app.providers_async import call_model
    from app.router_core import route_and_answer
    from app.judges import judge_answer
    from app.settings_dynamic import settings
    from app.utils.uncertainty import get_uncertainty_score
except ImportError:
    import sys
    sys.path.append(".")
    from app.providers_async import call_model
    from app.router_core import route_and_answer
    from app.judges import judge_answer
    from app.settings_dynamic import settings
    from app.utils.uncertainty import get_uncertainty_score

# ==============================================================================
# ⚙️ CONFIGURATION
# ==============================================================================
LOCAL_BASELINES = {
    "Local (Gemma 4B)": "ollama/gemma3:4b",
    "Local (Qwen 8B)": "ollama/qwen3:8b"  # Exemplo leve
    # "Local (Phi-4)": "ollama/phi4:14b" # Exemplo pesado
}
MODEL_SOTA = "openai/gpt-5.1"
OLLAMA_API_URL = os.getenv("OLLAMA_HOST", "http://ollama:11434")

PRICE_SOTA_INPUT = 0.005
PRICE_SOTA_OUTPUT = 0.015
RPM_OPENAI = 40

SAMPLES_PER_DATASET = 10 
NUM_RUNS = 3

# --- DEFINIÇÃO DE CONCORRÊNCIA ---
# Limites de Slots
SLOTS_LIGHT = 4  # Para modelos < 5GB VRAM
SLOTS_HEAVY = 1  # Para modelos > 10GB VRAM
SLOTS_API = 20   # Para GPT/Claude

# Lista de palavras-chave para identificar modelos leves
LIGHT_MODEL_KEYWORDS = ["gemma", "granite", "3b", "4b", "1b", "2b", "qwen:1.8b"]

OUTPUT_DIR = "thesis_results"
CHECKPOINT_FILE = f"{OUTPUT_DIR}/benchmark_checkpoint.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 🚦 RATE LIMITER (API)
# ==============================================================================
class RateLimiter:
    def __init__(self, max_calls_per_minute):
        self.max_calls = max_calls_per_minute
        self.period = 60.0
        self.timestamps = []
        self._lock = asyncio.Lock()

    async def wait(self):
        async with self._lock:
            now = time.time()
            self.timestamps = [t for t in self.timestamps if now - t < self.period]
            if len(self.timestamps) >= self.max_calls:
                sleep_time = self.timestamps[0] + self.period - now
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time + random.uniform(0.1, 0.5))
                now = time.time()
                self.timestamps = [t for t in self.timestamps if now - t < self.period]
            self.timestamps.append(time.time())

limiter_sota = RateLimiter(RPM_OPENAI)

# ==============================================================================
# 🛠️ UTILS
# ==============================================================================
@contextmanager
def temporary_setting(key, value):
    original_value = settings.get(key)
    try:
        settings.set(key, str(value), actor="benchmark_ablation")
        time.sleep(0.05)
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
                    await client.post(f"{OLLAMA_API_URL}/api/generate", json={"model": model['name'], "keep_alive": 0})
            await asyncio.sleep(1)
        except Exception: pass

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
                        if not any(m in [c['name'] for c in current] for m in models_to_kill): break
            
            await asyncio.sleep(1) # Cool down menor pois estamos em GPU dedicada (RTX)

            model_loaded = False
            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=600.0) as load_client:
                        resp = await load_client.post(
                            f"{OLLAMA_API_URL}/api/generate", 
                            json={"model": target_model_name.replace("ollama/", ""), "keep_alive": "60m"}
                        )
                        if resp.status_code == 200:
                            model_loaded = True
                            break
                        else: await asyncio.sleep(5)
                except Exception: await asyncio.sleep(5)
            
            if not model_loaded:
                logger.error(f"❌ CRITICAL: Could not load {target_model_name}")

        except Exception as e:
            logger.warning(f"⚠️ Ollama switch warning: {repr(e)}")

# ==============================================================================
# 📏 GROUND TRUTH & LOADERS
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
            try: return 1 if abs(float(nums[-1]) - float(ref.replace(",", ""))) < 1e-6 else 0
            except: pass
        return 0
    if dataset_name == "BBH-Date": return 1 if ref in pred else 0
    return 0 

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
    
    ans_local, meta_local = await call_model(local_model, query, max_tokens=512, temperature=0.1)
    
    judge_res = await judge_answer(query, ans_local)
    scores = [r["score"] for r in judge_res if "score" in r]
    score_local = sum(scores)/len(scores) if scores else 0.0
    
    if score_local >= 8.0:
        return {
            "answer": ans_local, "model": "FrugalGPT (Local)",
            "cost": meta_local.get("cost_per_1k", 0.0),
            "latency": time.time() - start_t, "load_time": meta_local.get("load_time", 0.0)
        }
    
    await limiter_sota.wait()
    ans_sota, meta_sota = await call_model(MODEL_SOTA, query, max_tokens=512, temperature=0.1)
    total_cost = meta_local.get("cost_per_1k", 0.0) + meta_sota.get("cost_per_1k", 0.0)
    
    return {
        "answer": ans_sota, "model": "FrugalGPT (SOTA)",
        "cost": total_cost, "latency": time.time() - start_t, "load_time": meta_local.get("load_time", 0.0)
    }

# ==============================================================================
# 🏃 EXECUTION ENGINE (DYNAMIC CONCURRENCY)
# ==============================================================================

def estimate_fallback_cost(query: str, answer: str) -> float:
    in_tokens = len(query) / 4
    out_tokens = len(answer) / 4
    return (in_tokens / 1000 * PRICE_SOTA_INPUT) + (out_tokens / 1000 * PRICE_SOTA_OUTPUT)

async def evaluate_interaction(mode_label: str, task: dict, run_id: int):
    query = task["query"]
    reference = task["reference"]
    dataset = task["dataset"]
    
    start_t = time.time()
    answer = ""
    model_used = "unknown"
    cost = 0.0
    latency = 0.0
    load_time = 0.0
    
    try:
        if mode_label == "Router (Hybrid)":
            res = await route_and_answer(query=query, use_cache=False, use_rag=True)
            answer = res.get("answer", "")
            model_used = res.get("model", "error")
            cost = res.get("cost_per_1k", 0.0)
            load_time = res.get("load_time_s", 0.0)

        elif mode_label == "FrugalGPT (Cascade)":
            res = await run_frugal_cascade(query)
            answer = res["answer"]
            model_used = res["model"]
            cost = res["cost"]
            total_latency = res["latency"]
            load_time = res["load_time"]
            if "SOTA" in model_used and cost <= 1e-6: cost += estimate_fallback_cost(query, answer)

        elif mode_label == "Router (No RAG)":
            res = await route_and_answer(query=query, use_cache=False, use_rag=False)
            answer = res.get("answer", "")
            model_used = res.get("model", "error")
            cost = res.get("cost_per_1k", 0.0)
            load_time = res.get("load_time_s", 0.0)

        elif mode_label == "Router (No Re-rank)":
            with temporary_setting("RERANK_ENABLED", "0"):
                res = await route_and_answer(query=query, use_cache=False, use_rag=True)
            answer = res.get("answer", "")
            model_used = res.get("model", "error")
            cost = res.get("cost_per_1k", 0.0)
            load_time = res.get("load_time_s", 0.0)

        elif mode_label == "Router (Random)":
            target = random.choice([list(LOCAL_BASELINES.values())[0], MODEL_SOTA])
            if "ollama" in target: await force_switch_ollama_model(target)
            answer, meta = await call_model(target, query, max_tokens=512, temperature=0.1)
            model_used = target
            cost = meta.get("cost_per_1k", 0.0)
            load_time = meta.get("load_time", 0.0)
            if cost <= 1e-6 and "gpt" in target: cost = estimate_fallback_cost(query, answer)

        elif mode_label == "SOTA (GPT-5.1)":
            await limiter_sota.wait()
            answer, meta = await call_model(MODEL_SOTA, query, max_tokens=512, temperature=0.1)
            model_used = MODEL_SOTA
            cost = meta.get("cost_per_1k", 0.0)
            if cost <= 1e-6: cost = estimate_fallback_cost(query, answer)

        elif mode_label in LOCAL_BASELINES:
            model_id = LOCAL_BASELINES[mode_label]
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
            "run_id": run_id, "id": task["id"], "dataset": dataset, "category": task["category"],
            "mode": mode_label, "model_used": model_used,
            "latency": effective_latency, "cost": cost, 
            "judge_score": judge_score, "is_correct": is_correct,
            "success": True,
            "query": query, "answer": answer, "reference": reference
        }

    except Exception as e:
        logger.error(f"Error in {mode_label}: {e}")
        return {
            "run_id": run_id, "id": task["id"], "dataset": dataset, "category": task["category"],
            "mode": mode_label, "model_used": "error",
            "latency": 0, "cost": 0, "judge_score": 0, "is_correct": 0,
            "success": False,
            "query": query, "answer": "", "reference": reference
        }

async def run_benchmark_suite():
    tasks_data = load_datasets()
    if not tasks_data: return pd.DataFrame()

    processed_keys = set()
    if os.path.exists(CHECKPOINT_FILE):
        logger.info(f"🔄 Checkpoint found: {CHECKPOINT_FILE}. Resuming...")
        try:
            df_existing = pd.read_csv(CHECKPOINT_FILE)
            for _, row in df_existing.iterrows():
                key = f"{row['run_id']}_{row['id']}_{row['mode']}"
                processed_keys.add(key)
            logger.info(f"✅ Loaded {len(processed_keys)} completed tasks.")
        except Exception as e:
            logger.error(f"⚠️ Failed to load checkpoint: {e}. Starting fresh.")

    modes = list(LOCAL_BASELINES.keys()) + [
        "SOTA (GPT-5.1)", 
        "FrugalGPT (Cascade)", 
        "Router (Hybrid)",
        "Router (No RAG)",
        "Router (No Re-rank)",
        "Router (Random)"
    ]
    
    # --- SEMÁFOROS DINÂMICOS ---
    sem_light = asyncio.Semaphore(SLOTS_LIGHT) # 4
    sem_heavy = asyncio.Semaphore(SLOTS_HEAVY) # 1
    sem_api   = asyncio.Semaphore(SLOTS_API)   # 20
    
    total_iterations = NUM_RUNS * len(tasks_data) * len(modes)
    pbar = tqdm(total=total_iterations, desc="Thesis Benchmark (Dynamic)")
    pbar.update(len(processed_keys))

    async def bounded_evaluate(mode, task, run):
        # Lógica de Seleção de Semáforo
        selected_sem = sem_heavy # Default seguro
        
        # 1. API (SOTA)
        if "SOTA" in mode or "GPT" in mode:
            selected_sem = sem_api
            
        # 2. Local Baselines
        elif mode in LOCAL_BASELINES:
            model_name = LOCAL_BASELINES[mode].lower()
            if any(k in model_name for k in LIGHT_MODEL_KEYWORDS):
                selected_sem = sem_light
            else:
                selected_sem = sem_heavy
                
        # 3. Router/Frugal (Híbridos)
        # Como eles usam o modelo local padrão (que geralmente é leve, ex: Gemma),
        # podemos ser um pouco mais permissivos, mas o Router pode chamar SOTA.
        # Estratégia segura: Tratar como "Leve" mas com cuidado.
        elif "Router" in mode or "Frugal" in mode:
            # Assume que o modelo local base é leve (Gemma 4B)
            selected_sem = sem_light 

        async with selected_sem:
            return await evaluate_interaction(mode, task, run)

    for run in range(1, NUM_RUNS + 1):
        for mode in modes:
            
            if mode in LOCAL_BASELINES:
                logger.info(f"🔄 Switching VRAM to {mode}...")
                await force_switch_ollama_model(LOCAL_BASELINES[mode])
            elif mode == "FrugalGPT (Cascade)" or "Router" in mode:
                default_local = list(LOCAL_BASELINES.values())[0]
                await force_switch_ollama_model(default_local)

            batch_tasks = []
            for task in tasks_data:
                unique_key = f"{run}_{task['id']}_{mode}"
                if unique_key not in processed_keys:
                    batch_tasks.append(bounded_evaluate(mode, task, run))
            
            if not batch_tasks: continue

            results_batch = await asyncio.gather(*batch_tasks)
            
            if results_batch:
                df_batch = pd.DataFrame(results_batch)
                header = not os.path.exists(CHECKPOINT_FILE)
                df_batch.to_csv(CHECKPOINT_FILE, mode='a', header=header, index=False)
                pbar.update(len(results_batch))

    pbar.close()
    
    if os.path.exists(CHECKPOINT_FILE):
        return pd.read_csv(CHECKPOINT_FILE)
    return pd.DataFrame()

# ==============================================================================
# 📊 PLOTTING
# ==============================================================================
def generate_meta_plots(df):
    if df.empty: return
    logger.info("📊 Generating plots...")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    palette = {
        "Local (Gemma 4B)": "#95a5a6", 
        "Local (Qwen 8B)": "#e67e22",
        "SOTA (GPT-5.1)": "#3498db", 
        "Router (Hybrid)": "#2ecc71", 
        "FrugalGPT (Cascade)": "#9b59b6",
        "Router (No RAG)": "#f39c12", 
        "Router (No Re-rank)": "#d35400", 
        "Router (Random)": "#e74c3c"
    }
    for m in df['mode'].unique():
        if m not in palette: palette[m] = "#34495e"

    # 1. Judge Reliability
    plt.figure(figsize=(8, 6))
    df_valid = df.dropna(subset=['is_correct'])
    df_valid['Ground Truth'] = df_valid['is_correct'].map({1: 'Correct', 0: 'Incorrect'})
    sns.boxplot(data=df_valid, x="Ground Truth", y="judge_score", hue="Ground Truth", palette="Set2", legend=False)
    plt.title("Judge Reliability: Score vs. Ground Truth", fontweight='bold')
    plt.savefig(f"{OUTPUT_DIR}/fig_judge_reliability.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Objective Accuracy
    plt.figure(figsize=(16, 6))
    sns.barplot(data=df_valid, x="dataset", y="is_correct", hue="mode", palette=palette, errorbar=None)
    plt.title("Objective Accuracy (Ground Truth) by Dataset", fontweight='bold')
    plt.ylabel("Accuracy (0.0 - 1.0)")
    plt.ylim(0, 1.0)
    plt.legend(title="Approach", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f"{OUTPUT_DIR}/fig_objective_accuracy.png", dpi=300, bbox_inches='tight')
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
    print("🧪 Starting Thesis Benchmark Suite (Dynamic Parallelism)...")
    df_results = asyncio.run(run_benchmark_suite())
    if not df_results.empty:
        generate_meta_plots(df_results)
        calculate_judge_metrics(df_results)
        print(f"\n🏆 Benchmark Complete. Check '{OUTPUT_DIR}' folder.")
    else:
        print("\n❌ Benchmark failed.")