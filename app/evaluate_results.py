# -*- coding: utf-8 -*-
"""
evaluate_results.py — Ultra-Optimized Phase 2: Batch Evaluation (Strict Evaluated Logic)
--------------------------------------------------------------------------------------
Otimizações: 
1. Lógica de Retomada Estrita: Baseia-se apenas na coluna 'evaluated'.
2. Ollama JSON Mode: Extração de dados estruturada e rápida.
3. Persistent HTTP Connection Pooling: Reuso de sockets para performance máxima.
4. Concorrência controlada: Otimizado para RTX 5070 Ti (8-12 workers).
"""

import pandas as pd
import asyncio
import httpx
import logging
import json
import os
import glob
import time
from tqdm.asyncio import tqdm

# ==============================================================================
# ⚙️ CONFIGURAÇÃO
# ==============================================================================
JUDGE_MODEL = "deepseek-r1:32b" 
OLLAMA_API_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
INPUT_DIR = "thesis_results"

# Ajuste de Performance para RTX 5070 Ti + 128GB RAM
JUDGE_CONCURRENCY = 8  
SAVE_INTERVAL = 20     # Salva o progresso no CSV a cada 20 avaliações

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("evaluator")

# ==============================================================================
# 🧠 NÚCLEO DE AVALIAÇÃO
# ==============================================================================

async def get_judge_score(client: httpx.AsyncClient, query, answer, reference):
    """
    Chama a API do Ollama forçando o modo JSON para obter o veredito técnico.
    """
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
    
    Return ONLY a JSON object with the following keys:
    "reasoning": (string) a very brief explanation of your decision.
    "score": (number) 0 or 10.
    """
    
    payload = {
        "model": JUDGE_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json", 
        "options": {
            "temperature": 0,
            "num_predict": 150, 
            "num_ctx": 4096
        }
    }
    
    try:
        resp = await client.post(f"{OLLAMA_API_URL}/api/generate", json=payload, timeout=120.0)
        
        if resp.status_code == 200:
            data = json.loads(resp.json().get("response", "{}"))
            return float(data.get("score", 0.0))
        else:
            logger.error(f"Erro na API Ollama: Status {resp.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Falha na comunicação: {str(e)}")
        return None

# ==============================================================================
# 🚀 PROCESSAMENTO EM LOTE
# ==============================================================================

async def process_batch():
    # 1. Localização do arquivo
    files = glob.glob(f"{INPUT_DIR}/raw_data_*.csv")
    if not files:
        logger.error(f"Nenhum arquivo CSV encontrado em {INPUT_DIR}")
        return
    
    latest_file = max(files, key=os.path.getctime)
    logger.info(f"📂 Carregando dados: {latest_file}")
    df = pd.read_csv(latest_file)
    
    # 2. Inicialização Estrita das Colunas
    if 'judge_score' not in df.columns:
        df['judge_score'] = 0.0
    
    if 'evaluated' not in df.columns:
        df['evaluated'] = False
    else:
        # Garante que valores nulos sejam tratados como False
        df['evaluated'] = df['evaluated'].fillna(False).astype(bool)

    # 3. Filtro de Pendentes (Independente da nota atual)
    # Só avaliamos se evaluated == False E se houver uma resposta para ler
    mask_pending = (df['evaluated'] == False) & (df['answer'].notna()) & (df['answer'] != "")
    pending_indices = df[mask_pending].index.tolist()

    if not pending_indices:
        logger.info("✅ Todas as linhas já estão marcadas como 'evaluated'.")
        return

    logger.info(f"🚀 Pendentes para avaliação: {len(pending_indices)}")

    # 4. Pool de Conexões e Execução
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=JUDGE_CONCURRENCY)
    async with httpx.AsyncClient(limits=limits, timeout=None) as client:
        
        # Warmup
        try:
            await client.post(f"{OLLAMA_API_URL}/api/generate", json={"model": JUDGE_MODEL, "keep_alive": -1})
        except: pass

        sem = asyncio.Semaphore(JUDGE_CONCURRENCY)

        async def worker(idx):
            async with sem:
                row = df.iloc[idx]
                score = await get_judge_score(client, row['query'], row['answer'], row['reference'])
                return idx, score

        tasks = [worker(i) for i in pending_indices]
        
        count = 0
        pbar = tqdm(total=len(tasks), desc="Judging")
        
        for coro in asyncio.as_completed(tasks):
            idx, score = await coro
            
            # Se o score for retornado (mesmo que seja 0), marcamos como avaliado
            if score is not None:
                df.at[idx, 'judge_score'] = score
                df.at[idx, 'evaluated'] = True
            
            count += 1
            pbar.update(1)

            # Checkpoint periódico
            if count % SAVE_INTERVAL == 0:
                df.to_csv(latest_file, index=False)

        pbar.close()

    # 5. Finalização
    df.to_csv(latest_file, index=False)
    
    output_file = latest_file.replace(".csv", "_evaluated.csv")
    df.to_csv(output_file, index=False)
    
    logger.info(f"✨ Concluído! Arquivo final: {output_file}")

# ==============================================================================
# ▶️ EXECUÇÃO
# ==============================================================================

if __name__ == "__main__":
    start_time = time.perf_counter()
    try:
        asyncio.run(process_batch())
    except KeyboardInterrupt:
        logger.info("\n🛑 Interrompido! O progresso foi salvo.")
    except Exception as e:
        logger.error(f"🔥 Erro fatal: {e}")
    finally:
        end_time = time.perf_counter()
        logger.info(f"⏱️ Tempo total: {end_time - start_time:.2f}s")