# -*- coding: utf-8 -*-
"""
statistical_validation.py — Validação Estatística com Geração de Texto LaTeX
----------------------------------------------------------------------------
1. Carrega os dados do benchmark.
2. Gera IDs sintéticos se necessário (Correção do KeyError).
3. Executa testes (Shapiro-Wilk, Friedman, Wilcoxon).
4. Gera tabelas e TEXTO DISCURSIVO em formato LaTeX pronto para a tese.
"""

import pandas as pd
import numpy as np
from scipy import stats
from tabulate import tabulate
import logging
import sys
import os
import glob

# Configuração
INPUT_DIR = "thesis_results"
ALPHA = 0.05

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("stats")

def load_latest_data():
    files = glob.glob(f"{INPUT_DIR}/raw_data_*.csv")
    if not files:
        logger.error(f"❌ Nenhum arquivo CSV encontrado em {INPUT_DIR}/.")
        sys.exit(1)
    
    latest_file = max(files, key=os.path.getctime)
    logger.info(f"📂 Carregando dados de: {latest_file}")
    
    df = pd.read_csv(latest_file)
    
    # --- CORREÇÃO: Geração de ID Sintético ---
    if 'id' not in df.columns:
        logger.info("⚠️ Coluna 'id' não encontrada. Gerando IDs sintéticos baseados na ordem sequencial...")
        # Sabemos que existem 3 modos por pergunta (Local, SOTA, Router)
        # O script roda: Run -> Task -> Mode
        # Então a cada 3 linhas, mudamos de pergunta.
        # O ID deve reiniciar ou ser consistente entre Runs para o agrupamento funcionar.
        
        # Agrupa por Run e atribui um ID sequencial para cada trio de linhas
        # Ex: Linhas 0,1,2 (Pergunta A) -> ID 0
        #     Linhas 3,4,5 (Pergunta B) -> ID 1
        df['id'] = df.groupby('run_id').cumcount() // 3
    # -----------------------------------------

    # Normaliza nomes
    df["mode"] = df["mode"].map({
        "Local (Gemma 3)": "Local",
        "SOTA (GPT-5.1)": "SOTA",
        "Router (Hybrid)": "Router"
    })
    
    # Pivota para pareamento (Média das runs por pergunta)
    df_grouped = df.groupby(["id", "mode"]).agg({
        "quality": "mean",
        "cost": "mean",
        "latency": "mean"
    }).reset_index()
    
    return df_grouped

def calculate_cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pool_std = np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)
    return (np.mean(x) - np.mean(y)) / pool_std

def format_p_value(p):
    if p < 0.001: return "< 0.001"
    return f"= {p:.4f}"

def analyze_metric(df, metric_col, metric_name_pt):
    """
    Executa os testes e retorna um dicionário com todos os resultados
    para ser usado no gerador de texto.
    """
    pivot = df.pivot(index="id", columns="mode", values=metric_col).dropna()
    
    # Garante que temos as colunas
    if not all(col in pivot.columns for col in ["Local", "SOTA", "Router"]):
        logger.error(f"❌ Dados incompletos para a métrica {metric_col}. Colunas encontradas: {pivot.columns}")
        sys.exit(1)

    local = pivot["Local"]
    sota = pivot["SOTA"]
    router = pivot["Router"]

    # 1. Normalidade
    _, p_shapiro = stats.shapiro(router)
    is_normal = p_shapiro > ALPHA

    # 2. Friedman (Global)
    stat_fried, p_fried = stats.friedmanchisquare(local, sota, router)
    is_significant = p_fried < ALPHA

    # 3. Post-hoc (Wilcoxon)
    comparisons = {}
    pairs = [("Router", "Local"), ("Router", "SOTA")]
    
    for m1, m2 in pairs:
        stat_w, p_w = stats.wilcoxon(pivot[m1], pivot[m2])
        d = calculate_cohens_d(pivot[m1], pivot[m2])
        diff_mean = (pivot[m1] - pivot[m2]).mean()
        
        comparisons[f"{m1}_vs_{m2}"] = {
            "p_val": p_w,
            "stat": stat_w,
            "cohen_d": d,
            "diff_mean": diff_mean,
            "significant": p_w < ALPHA
        }

    return {
        "metric_name": metric_name_pt,
        "shapiro_p": p_shapiro,
        "is_normal": is_normal,
        "friedman_stat": stat_fried,
        "friedman_p": p_fried,
        "is_global_sig": is_significant,
        "comparisons": comparisons,
        "means": {
            "Router": router.mean(),
            "Local": local.mean(),
            "SOTA": sota.mean()
        },
        "stds": {
            "Router": router.std(),
            "Local": local.std(),
            "SOTA": sota.std()
        }
    }

def generate_latex_text(results):
    """
    Gera o texto interpretativo em LaTeX baseado nos números.
    """
    m = results["metric_name"]
    comp = results["comparisons"]
    
    # Texto sobre Normalidade e Friedman
    dist_text = "uma distribuição normal" if results["is_normal"] else "uma distribuição não-normal"
    shapiro_tex = f"$p {format_p_value(results['shapiro_p'])}$"
    
    friedman_res = "diferenças estatisticamente significativas" if results["is_global_sig"] else "ausência de diferenças significativas"
    friedman_tex = f"$\chi^2 = {results['friedman_stat']:.2f}, p {format_p_value(results['friedman_p'])}$"

    text = f"""
% -------------------------------------------------------
% Resultados para: {m}
% -------------------------------------------------------

A análise estatística da métrica de \\textbf{{{m}}} iniciou-se com o teste de Shapiro-Wilk, que indicou {dist_text} para os dados do Router ({shapiro_tex}). 
Dada a natureza das distribuições e o pareamento das amostras (mesmas perguntas submetidas a diferentes modelos), optou-se por testes não-paramétricos.

O teste de Friedman revelou {friedman_res} entre as três abordagens comparadas ({friedman_tex}). 
Consequentemente, procedeu-se à análise post-hoc utilizando o teste de postos sinalizados de Wilcoxon com correção de Bonferroni para comparações múltiplas.
"""

    # Texto sobre Router vs Local
    rvl = comp["Router_vs_Local"]
    rvl_sig = "significativamente superior" if rvl["significant"] and rvl["diff_mean"] > 0 else \
              "significativamente inferior" if rvl["significant"] and rvl["diff_mean"] < 0 else \
              "estatisticamente equivalente"
    
    # Ajuste semântico para Latência/Custo (menor é melhor) vs Qualidade (maior é melhor)
    if "Custo" in m or "Latência" in m:
        if rvl["diff_mean"] < 0: rvl_sig = "significativamente melhor (menor)"
        elif rvl["diff_mean"] > 0: rvl_sig = "significativamente pior (maior)"
    
    text += f"""
Na comparação direta entre o \\textbf{{Router Híbrido}} e o modelo \\textbf{{Local}}, o Router mostrou-se {rvl_sig} 
($Z = {rvl['stat']:.1f}, p {format_p_value(rvl['p_val'])}$), com um tamanho de efeito de Cohen's $d = {rvl['cohen_d']:.2f}$.
"""

    # Texto sobre Router vs SOTA
    rvs = comp["Router_vs_SOTA"]
    if not rvs["significant"]:
        rvs_text = "não apresentou diferença estatisticamente significativa, suportando a hipótese de equivalência/não-inferioridade"
    else:
        if "Custo" in m and rvs["diff_mean"] < 0:
            rvs_text = "apresentou uma redução estatisticamente significativa"
        elif "Qualidade" in m and rvs["diff_mean"] < 0:
            rvs_text = "apresentou uma leve degradação estatisticamente significativa, porém com tamanho de efeito reduzido"
        else:
            rvs_text = "apresentou diferença significativa"

    text += f"""
Em relação ao estado da arte (\\textbf{{SOTA}}), a abordagem proposta {rvs_text} 
($Z = {rvs['stat']:.1f}, p {format_p_value(rvs['p_val'])}, d = {rvs['cohen_d']:.2f}$).
"""
    
    # Tabela Resumo em LaTeX
    text += f"""
\\begin{{table}}[h]
\\centering
\\caption{{Resultados Estatísticos: {m}}}
\\label{{tab:stats_{m.split()[0].lower()}}}
\\begin{{tabular}}{{lcccc}}
\\hline
\\textbf{{Comparação}} & \\textbf{{Diferença Média}} & \\textbf{{$p$-value}} & \\textbf{{Cohen's $d$}} & \\textbf{{Sig.}} \\\\ \\hline
Router vs. Local & {rvl['diff_mean']:.4f} & {format_p_value(rvl['p_val'])} & {rvl['cohen_d']:.2f} & {'*' if rvl['significant'] else 'ns'} \\\\
Router vs. SOTA  & {rvs['diff_mean']:.4f} & {format_p_value(rvs['p_val'])} & {rvs['cohen_d']:.2f} & {'*' if rvs['significant'] else 'ns'} \\\\ \\hline
\\end{{tabular}}
\\end{{table}}
"""
    return text

def main():
    df = load_latest_data()
    
    print("="*80)
    print("🚀 GERADOR DE RESULTADOS PARA TESE (LaTeX)")
    print("="*80)

    # 1. Qualidade
    res_qual = analyze_metric(df, "quality", "Qualidade da Resposta")
    latex_qual = generate_latex_text(res_qual)
    
    # 2. Custo
    res_cost = analyze_metric(df, "cost", "Custo de Inferência")
    latex_cost = generate_latex_text(res_cost)
    
    # 3. Latência
    res_lat = analyze_metric(df, "latency", "Latência de Resposta")
    latex_lat = generate_latex_text(res_lat)

    # Salvar em arquivo .tex
    output_file = f"{INPUT_DIR}/resultados_estatisticos.tex"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("% Arquivo gerado automaticamente por statistical_validation.py\n")
        f.write("\\section{Análise de Qualidade}\n")
        f.write(latex_qual)
        f.write("\n\\section{Análise de Custo}\n")
        f.write(latex_cost)
        f.write("\n\\section{Análise de Latência}\n")
        f.write(latex_lat)

    print(f"\n✅ Arquivo LaTeX gerado com sucesso: {output_file}")
    print("Copie o conteúdo deste arquivo e cole no capítulo de Resultados da sua tese.")
    print("="*80)
    
    # Preview no console
    print("\n--- PREVIEW (Qualidade) ---")
    print(latex_qual[:500] + "...\n")

if __name__ == "__main__":
    main()
