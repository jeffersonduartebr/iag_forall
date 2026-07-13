# -*- coding: utf-8 -*-
# Objective: Application-side script for statistical validation.
"""
statistical_validation.py — Validação Estatística (Sincronizado com Benchmark Final)
------------------------------------------------------------------------------------
1. Carrega dados (suporta FrugalGPT e Ablações).
2. Gera IDs sintéticos.
3. Executa testes (Friedman/Wilcoxon).
4. Gera LaTeX ordenado: Local -> Frugal -> SOTA -> Router -> Ablações.
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging
import sys
import os
import glob

# Configuração
INPUT_DIR = "thesis_results"
ALPHA = 0.05
# O nome do modelo principal no CSV gerado pelo benchmark é "Router (Hybrid)"
TARGET_MODEL_CSV_NAME = "Router (Hybrid)" 

# Mapeamento para nomes bonitos no LaTeX
NAME_MAP = {
    "Local (Gemma 4B)": "Gemma-4B",
    "Local (Qwen 8B)": "Qwen-8B",
    "SOTA (GPT-5.1)": "GPT-5.1",
    "FrugalGPT (Cascade)": "FrugalGPT",
    "Router (Hybrid)": "Router (Proposed)", # Nome de destaque na tese
    "Router (No RAG)": "Ablation-NoRAG",
    "Router (No Re-rank)": "Ablation-NoRerank",
    "Router (Random)": "Ablation-Random"
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("stats")

def load_latest_data():
    """Carrega latest data."""
    files = glob.glob(f"{INPUT_DIR}/*data_*.csv")
    if not files:
        logger.error(f"❌ No CSV files found in {INPUT_DIR}/.")
        sys.exit(1)
    
    latest_file = max(files, key=os.path.getctime)
    logger.info(f"📂 Loading data from: {latest_file}")
    
    df = pd.read_csv(latest_file)
    
    # Normaliza nomes
    df["mode_clean"] = df["mode"].apply(lambda x: NAME_MAP.get(x, x))
    
    # Garante ID
    if 'id' not in df.columns:
        num_modes = df['mode'].nunique()
        df['id'] = df.groupby('run_id').cumcount() // num_modes

    # Agregação
    agg_dict = {"quality": "mean", "cost": "mean", "latency": "mean"}
    if "is_correct" in df.columns:
        agg_dict["is_correct"] = "mean"

    df_grouped = df.groupby(["id", "mode_clean"]).agg(agg_dict).reset_index()
    return df_grouped

def calculate_cohens_d(x, y):
    """Execute the calculate cohens d routine.

This helper encapsulates one focused step used by the surrounding workflow."""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pool_std = np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)
    if pool_std == 0: return 0.0
    return (np.mean(x) - np.mean(y)) / pool_std

def format_p_value(p):
    """Execute the format p value routine.

This helper encapsulates one focused step used by the surrounding workflow."""
    if p < 0.001: return "< 0.001"
    return f"= {p:.4f}"

def analyze_metric(df, metric_col, metric_name_en):
    """Execute the analyze metric routine.

This helper encapsulates one focused step used by the surrounding workflow."""
    if metric_col not in df.columns: return None

    pivot = df.pivot(index="id", columns="mode_clean", values=metric_col).dropna()
    
    # Define o alvo (Router)
    target = NAME_MAP.get(TARGET_MODEL_CSV_NAME, "Router")
    
    if target not in pivot.columns:
        logger.error(f"❌ Target '{target}' not found in columns: {pivot.columns}")
        return None

    baselines = [c for c in pivot.columns if c != target]
    
    # 1. Normality
    try:
        _, p_shapiro = stats.shapiro(pivot[target])
        is_normal = p_shapiro > ALPHA
    except: is_normal = False

    # 2. Friedman
    try:
        all_groups = [pivot[c] for c in pivot.columns]
        stat_fried, p_fried = stats.friedmanchisquare(*all_groups)
        is_global_sig = p_fried < ALPHA
    except ValueError:
        p_fried, is_global_sig = 1.0, False

    # 3. Post-hoc
    comparisons = {}
    for base in baselines:
        try:
            stat_w, p_w = stats.wilcoxon(pivot[target], pivot[base])
            d = calculate_cohens_d(pivot[target], pivot[base])
            diff_mean = (pivot[target] - pivot[base]).mean()
            sig = p_w < ALPHA
        except ValueError:
            p_w, d, diff_mean, sig = 1.0, 0.0, 0.0, False
        
        comparisons[base] = {
            "p_val": p_w, "stat": stat_w, "cohen_d": d,
            "diff_mean": diff_mean, "significant": sig
        }

    return {
        "metric_name": metric_name_en,
        "target": target,
        "shapiro_p": p_shapiro if 'p_shapiro' in locals() else 0,
        "is_normal": is_normal,
        "friedman_stat": stat_fried if 'stat_fried' in locals() else 0,
        "friedman_p": p_fried,
        "is_global_sig": is_global_sig,
        "comparisons": comparisons,
        "baselines": baselines
    }

def generate_latex_text(results):
    """Execute the generate latex text routine.

This helper encapsulates one focused step used by the surrounding workflow."""
    if not results: return ""
    
    m = results["metric_name"]
    target = results["target"]
    
    text = f"""
% Analysis: {m}
The analysis of \\textbf{{{m}}} compared \\textit{{{target}}} against baselines. 
Friedman test: $\chi^2 = {results['friedman_stat']:.2f}, p {format_p_value(results['friedman_p'])}$.

Post-hoc Wilcoxon results:
\\begin{{itemize}}
"""
    
    table_rows = []
    
    # Ordenação Lógica para Tabela: Local -> Frugal -> SOTA -> Ablações
    def sort_key(name):
        """Execute the sort key routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        if "Gemma" in name or "Qwen" in name: return 0
        if "Frugal" in name: return 1
        if "GPT" in name or "SOTA" in name: return 2
        return 3 # Ablações
        
    sorted_baselines = sorted(results["baselines"], key=sort_key)

    for base in sorted_baselines:
        comp = results["comparisons"][base]
        
        if not comp["significant"]:
            interp = "equivalent"
        else:
            is_higher = comp["diff_mean"] > 0
            if "Quality" in m or "Accuracy" in m:
                interp = "superior" if is_higher else "inferior"
            else:
                interp = "worse (higher)" if is_higher else "better (lower)"

        text += f"    \\item vs. \\textbf{{{base}}}: {interp} ($p {format_p_value(comp['p_val'])}, d={comp['cohen_d']:.2f}$).\n"
        
        sig_mark = '*' if comp['significant'] else 'ns'
        table_rows.append([
            f"{target} vs. {base}",
            f"{comp['diff_mean']:.4f}",
            f"{format_p_value(comp['p_val'])}",
            f"{comp['cohen_d']:.2f}",
            sig_mark
        ])

    text += "\\end{itemize}\n"
    
    text += f"""
\\begin{{table}}[H]
\\centering
\\caption{{Statistical Comparison: {m}}}
\\label{{tab:stats_{m.split()[0].lower()}}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Comparison}} & \\textbf{{Mean Diff.}} & \\textbf{{$p$-value}} & \\textbf{{Cohen's $d$}} & \\textbf{{Sig.}} \\\\ \\midrule
"""
    for row in table_rows:
        text += f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} \\\\ \n"
    text += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    
    return text

def main():
    """Execute the main routine.

This helper encapsulates one focused step used by the surrounding workflow."""
    print("="*80)
    print("🚀 THESIS STATISTICAL GENERATOR (Final Sync)")
    print("="*80)
    
    df = load_latest_data()
    print(f"ℹ️  Models found: {df['mode_clean'].unique()}")
    
    sections = []
    # Ordem de importância
    metrics = [
        ("is_correct", "Objective Accuracy"),
        ("cost", "Inference Cost"),
        ("latency", "Latency"),
        ("quality", "Judge Quality Score")
    ]
    
    for col, name in metrics:
        res = analyze_metric(df, col, name)
        if res: sections.append(generate_latex_text(res))

    output_file = f"{INPUT_DIR}/statistical_report_final.tex"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("% Generated by statistical_validation.py\n")
        for sec in sections:
            f.write(sec + "\n")

    print(f"\n✅ LaTeX file generated: {output_file}")

if __name__ == "__main__":
    main()