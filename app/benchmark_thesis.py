# -*- coding: utf-8 -*-
"""
benchmark_thesis.py — Phase 1: Generation & Performance (NO JUDGE)
------------------------------------------------------------------
Executes the comparative study focusing on Latency, Cost, and Ground Truth.
Skips the LLM-as-a-Judge step to maximize GPU throughput for inference.

Output: 'raw_data_*.csv' (to be consumed by evaluate_results.py)
"""

import asyncio
import time
import os
import logging
import pandas as pd
import numpy as np
import httpx
import re
import random
from contextlib import contextmanager
from datasets import load_dataset
from tqdm.asyncio import tqdm
from datetime import datetime

# --- System Imports ---
try:
    from app.providers_async import call_model
    from app.router_core import route_and_answer
    from app.settings_dynamic import settings
    from app.utils.uncertainty import get_uncertainty_score
except ImportError:
    import sys
    sys.path.append(".")
    from app.providers_async import call_model
    from app.router_core import route_and_answer
    from app.settings_dynamic import settings
    from app.utils.uncertainty import get_uncertainty_score

# ==============================================================================
# ⚙️ CONFIGURATION
# ==============================================================================
LOCAL_BASELINES = {
    "Local (Gemma 4B)": "ollama/gemma3:4b",
    # "Local (Qwen 8B)": "ollama/qwen3:8b" 
}
MODEL_SOTA = "openai/gpt-5.2"
OLLAMA_API_URL = os.getenv("OLLAMA_HOST", "http://ollama:11434")

PRICE_SOTA_INPUT = 0.005
PRICE_SOTA_OUTPUT = 0.015
RPM_OPENAI = 40

SAMPLES_PER_DATASET = 75 
NUM_RUNS = 15
CONCURRENCY_LIMIT = 18

OUTPUT_DIR = "thesis_results"
CHECKPOINT_FILE = f"{OUTPUT_DIR}/benchmark_checkpoint.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 🚦 RATE LIMITER
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
                    # Polling rápido
                    for _ in range(5):
                        await asyncio.sleep(1)
                        check = await client.get(f"{OLLAMA_API_URL}/api/ps")
                        current = check.json().get('models', [])
                        if not any(m in [c['name'] for c in current] for m in models_to_kill): break
            
            # Warmup
            async with httpx.AsyncClient(timeout=300.0) as load_client:
                await load_client.post(
                    f"{OLLAMA_API_URL}/api/generate", 
                    json={"model": target_model_name.replace("ollama/", ""), "keep_alive": "60m"}
                )
        except Exception as e:
            logger.warning(f"⚠️ Ollama switch warning: {repr(e)}")

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
            try: return 1 if abs(float(nums[-1]) - float(ref.replace(",", ""))) < 1e-6 else 0
            except: pass
        return 0
    if dataset_name == "BBH-Date": return 1 if ref in pred else 0
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
    
    ans_local, meta_local = await call_model(local_model, query, max_tokens=512, temperature=0.1)
    
    # Simulação de Juiz Rápido (Heurística) para decisão de cascata
    # Na Fase 1, não usamos LLM Judge. Usamos tamanho/certeza ou Ground Truth se disponível (cheat)
    # Para ser justo, vamos usar uma heurística de tamanho + palavras-chave de incerteza
    score_local = 5.0
    if len(ans_local) > 50 and "I don't know" not in ans_local:
        score_local = 8.0 # Passa
    
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
# 🏃 EXECUTION ENGINE
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
    uncertainty = 0.0
    
    try:
        if mode_label == "Router (Hybrid)":
            res = await route_and_answer(query=query, use_cache=False, use_rag=True)
            answer = res.get("answer", "")
            model_used = res.get("model", "error")
            cost = res.get("cost_per_1k", 0.0)
            load_time = res.get("load_time_s", 0.0)
            uncertainty = res.get("route", {}).get("objectives", {}).get("uncertainty", 0.0)

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
            if "gpt" in target: await limiter_sota.wait()
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
            uncertainty = await asyncio.to_thread(get_uncertainty_score, query, "text")

        elif mode_label in LOCAL_BASELINES:
            model_id = LOCAL_BASELINES[mode_label]
            answer, meta = await call_model(model_id, query, max_tokens=512, temperature=0.1)
            model_used = model_id
            cost = meta.get("cost_per_1k", 0.0)
            load_time = meta.get("load_time", 0.0)
            uncertainty = await asyncio.to_thread(get_uncertainty_score, query, "text")
            
        else:
            raise ValueError(f"Modo desconhecido: {mode_label}")
        
        if mode_label != "FrugalGPT (Cascade)":
            total_latency = time.time() - start_t
        
        effective_latency = max(0.0, total_latency - load_time)

        # --- 2. Ground Truth (Rápido) ---
        is_correct = check_correctness(dataset, answer, reference)

        # --- 3. Judge Score (PULADO - Fase 2) ---
        judge_score = 0.0 

        return {
            "run_id": run_id, "id": task["id"], "dataset": dataset, "category": task["category"],
            "mode": mode_label, "model_used": model_used,
            "latency": effective_latency, "cost": cost, 
            "judge_score": judge_score, "is_correct": is_correct,
            "uncertainty": uncertainty,
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
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    total_iterations = NUM_RUNS * len(tasks_data) * len(modes)
    pbar = tqdm(total=total_iterations, desc="Thesis Benchmark (Phase 1)")
    pbar.update(len(processed_keys))

    async def bounded_evaluate(mode, task, run):
        async with semaphore:
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

if __name__ == "__main__":
    print("🧪 Starting Thesis Benchmark Suite (Phase 1: Generation)...")
    df_results = asyncio.run(run_benchmark_suite())
    if not df_results.empty:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        final_file = f"{OUTPUT_DIR}/raw_data_{timestamp}.csv"
        df_results.to_csv(final_file, index=False)
        print(f"\n🏆 Phase 1 Complete. Data saved to '{final_file}'.")
        print("👉 Now run 'python evaluate_results.py' for Phase 2 (Judging).")
    else:
        print("\n❌ Benchmark failed.")
