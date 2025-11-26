# -*- coding: utf-8 -*-
"""
calculate_savings.py
--------------------
Gera relatório de economia financeira comparando o Router Híbrido
com o Baseline de Mercado (GPT-5 Standard).
"""

import os
import pandas as pd
from sqlalchemy import create_engine
from tabulate import tabulate

# --- Configuração do Baseline (GPT-5 Standard) ---
# Valores: Input $1.25/1M, Output $10.00/1M
BASELINE_NAME = "OpenAI GPT-5"
PRICE_IN_1M = 1.25
PRICE_OUT_1M = 10.00

# Conexão
DB_USER = os.getenv("DB_USER", "router_user")
DB_PASS = os.getenv("DB_PASS", "router_pass")
DB_HOST = os.getenv("DB_HOST", "mariadb") 
DB_NAME = os.getenv("DB_NAME", "routerdb")
DB_PORT = 3307 if DB_HOST == "localhost" else 3306

DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def run_analysis():
    print(f"🔌 Conectando ao banco {DB_NAME} em {DB_HOST}...")
    engine = create_engine(DB_URL)

    # 1. Carregar dados brutos
    # Nota: No router_core novo, 'cost_per_1k' guarda o custo TOTAL da transação ($), não rate.
    query = """
    SELECT 
        id, chosen_model, modality,
        LENGTH(query_text) / 4 as tokens_in_est, -- Estimativa caso não tenha logado token
        LENGTH(answer) / 4 as tokens_out_est,
        cost_per_1k as actual_cost,
        quality, created_at
    FROM query_log
    WHERE created_at > NOW() - INTERVAL 30 DAY
    """
    
    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        print(f"❌ Erro ao ler banco: {e}")
        return

    if df.empty:
        print("⚠️ Nenhum dado encontrado no log.")
        return

    print(f"📊 Analisando {len(df)} requisições contra Baseline: {BASELINE_NAME}...\n")

    # 2. Calcular Custo do Baseline (Hipotético GPT-5)
    df['baseline_cost'] = (
        (df['tokens_in_est'] / 1_000_000 * PRICE_IN_1M) + 
        (df['tokens_out_est'] / 1_000_000 * PRICE_OUT_1M)
    )
    
    # Custo extra para visão no baseline (aprox)
    df.loc[df['modality'] == 'vision', 'baseline_cost'] += 0.004 

    # 3. Métricas Finais
    total_actual = df['actual_cost'].sum()
    total_baseline = df['baseline_cost'].sum()
    savings = total_baseline - total_actual
    savings_pct = (savings / total_baseline) * 100 if total_baseline > 0 else 0

    # 4. Agrupamento
    stats = df.groupby('chosen_model').agg(
        count=('id', 'count'),
        avg_quality=('quality', 'mean'),
        total_cost=('actual_cost', 'sum'),
        potential_cost_gpt5=('baseline_cost', 'sum')
    ).reset_index()
    
    stats['savings'] = stats['potential_cost_gpt5'] - stats['total_cost']

    # --- RELATÓRIO ---
    print("="*70)
    print(f"💰 RELATÓRIO DE ECONOMIA FINANCEIRA (vs {BASELINE_NAME})")
    print("="*70)
    print(f"Total de Queries:       {len(df)}")
    print(f"Custo Baseline (GPT-5): ${total_baseline:.4f}")
    print(f"Custo Real (Router):    ${total_actual:.4f}")
    print("-" * 70)
    print(f"💸 ECONOMIA TOTAL:       ${savings:.4f}")
    print(f"📉 REDUÇÃO DE CUSTO:     {savings_pct:.2f}%")
    print("="*70)
    
    print("\nDetalhe por Modelo Escolhido:")
    print(tabulate(stats, headers='keys', tablefmt='psql', floatfmt=".4f"))

if __name__ == "__main__":
    run_analysis()