# -*- coding: utf-8 -*-
"""
evaluate_results.py — Phase 2: Batch Evaluation (Parallelized)
--------------------------------------------------------------
Executes LLM-as-a-Judge on the generated CSV data.
Optimized for high throughput on RTX 5070 Ti + 128GB RAM.
"""

import pandas as pd
import asyncio
import httpx
import logging
import re
import os
import glob
from tqdm.asyncio import tqdm

# --- CONFIGURAÇÃO ---
# Modelo Juiz (Deve ser forte: Qwen 32B, Llama 70B ou DeepSeek R1)
JUDGE_MODEL = "ollama/deepseek-r1:32b" 
OLLAMA_API_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
INPUT_DIR = "thesis_results"

# Aumente para 8 ou 10 se tiver configurado OLLAMA_NUM_PARALLEL=8
JUDGE_CONCURRENCY = 4 

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("evaluator")

async def get_judge_score(query, answer, reference):
    """
    Chama o Ollama para julgar. Suporta extração de <think> e Score.
    """
    # Prompt otimizado para extração robusta
    prompt = f"""
    You are an impartial technical judge. Evaluate the AI response based on the Ground Truth.
    
    QUESTION: {query}
    GROUND TRUTH: {reference}
    AI RESPONSE: {answer}
    
    INSTRUCTIONS:
    1. Compare the AI RESPONSE with the GROUND TRUTH.
    2. Check for factual correctness and logic.
    3. If the answer is correct according to the ground truth, score 10.
    4. If it contradicts the ground truth or is wrong, score 0.
    
    OUTPUT FORMAT:
    Reasoning: <brief explanation>
    Score: <0 or 10>
    """
    
    payload = {
        "model": JUDGE_MODEL.replace("ollama/", ""),
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0, 
            "num_predict": 512,
            "num_ctx": 4096 # Garante contexto suficiente para input + output
        }
    }
    
    # Timeout generoso para modelos grandes rodando em paralelo
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            resp = await client.post(f"{OLLAMA_API_URL}/api/generate", json=payload)
            if resp.status_code != 200: 
                logger.warning(f"Judge API Error: {resp.status_code} - {resp.text}")
                return 0.0
            
            text = resp.json().get("response", "")
            
            # Remove bloco de pensamento <think> se existir (DeepSeek/Phi-4)
            text_clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            
            # Extrai nota (procura pelo último número após "Score:")
            match = re.findall(r"Score:\s*(\d+(?:\.\d+)?)", text_clean, re.IGNORECASE)
            if match:
                val = float(match[-1])
                # Normaliza se o modelo alucinar e der nota fora de 0-10
                return max(0.0, min(10.0, val))
            
            # Fallback: Procura apenas números isolados no final
            # Útil se o modelo responder apenas "10"
            if text_clean.strip().isdigit():
                return float(text_clean.strip())
                
            return 0.0
        except Exception as e:
            logger.error(f"Judge exception: {e}")
            return 0.0

async def process_batch():
    # 1. Carrega o CSV mais recente
    files = glob.glob(f"{INPUT_DIR}/raw_data_*.csv")
    if not files:
        logger.error("No CSV found.")
        return
    
    latest_file = max(files, key=os.path.getctime)
    logger.info(f"📂 Processing: {latest_file}")
    df = pd.read_csv(latest_file)
    
    # Verifica se já tem notas parciais (Resume)
    if 'judge_score' not in df.columns:
        df['judge_score'] = 0.0
        
    # Identifica linhas que precisam de avaliação
    # (Score 0 pode ser nota real ou não avaliado. 
    #  Melhor critério: se já rodou, não roda de novo, a menos que queira reavaliar zeros)
    # Aqui assumimos: Se judge_score é 0 E a resposta não é vazia, vamos reavaliar 
    # (ou você pode criar uma coluna 'evaluated' para controle explícito)
    
    # Para simplificar: Avalia tudo que ainda não foi processado na Fase 1
    # Na Fase 1, judge_score foi setado como 0.0.
    # Vamos processar todas as linhas.
    
    # 2. Prepara o Modelo do Juiz (Warmup)
    model_name = JUDGE_MODEL.replace("ollama/", "")
    logger.info(f"🧠 Loading Judge: {model_name}...")
    async with httpx.AsyncClient(timeout=600.0) as client:
        await client.post(f"{OLLAMA_API_URL}/api/pull", json={"name": model_name})
        await client.post(f"{OLLAMA_API_URL}/api/generate", json={"model": model_name, "keep_alive": -1})

    # 3. Loop de Avaliação Paralela
    sem = asyncio.Semaphore(JUDGE_CONCURRENCY)
    
    async def evaluate_row(index, row):
        async with sem:
            # Se a resposta for vazia (erro na inferência), nota é 0
            if pd.isna(row['answer']) or str(row['answer']).strip() == "":
                return index, 0.0
            
            score = await get_judge_score(row['query'], row['answer'], row['reference'])
            return index, score

    tasks = []
    for idx, row in df.iterrows():
        tasks.append(evaluate_row(idx, row))
    
    logger.info(f"🚀 Evaluating {len(tasks)} responses with concurrency={JUDGE_CONCURRENCY}...")
    
    # Barra de progresso
    counter = 0
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Judging"):
        idx, score = await f
        df.at[idx, 'judge_score'] = score
        
        counter += 1
        # Salva checkpoint a cada 50 avaliações
        if counter % 50 == 0:
            df.to_csv(latest_file, index=False)

    # 5. Salva Final
    output_file = latest_file.replace(".csv", "_evaluated.csv")
    df.to_csv(output_file, index=False)
    logger.info(f"✅ Done! Saved to {output_file}")
    
    # Atualiza o arquivo original também para facilitar scripts subsequentes
    df.to_csv(latest_file, index=False)

if __name__ == "__main__":
    asyncio.run(process_batch())