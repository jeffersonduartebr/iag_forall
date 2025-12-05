# -*- coding: utf-8 -*-
"""
statistical_validation.py — Universal Statistical Validation (Thesis Edition)
-----------------------------------------------------------------------------
1. Loads benchmark data (supports multiple baselines including FrugalGPT).
2. Generates synthetic IDs if necessary for paired testing.
3. Executes Friedman (Global) and Wilcoxon (Post-hoc Router vs All) tests.
4. Generates LaTeX text in formal American English.
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging
import sys
import os
import glob

# Configuration
INPUT_DIR = "thesis_results"
ALPHA = 0.05
TARGET_MODEL_KEY = "Router" # Keyword to identify the proposed solution

# Mapping for clean LaTeX names
NAME_MAP = {
    "Local (Gemma 4B)": "Gemma-4B",
    "Local (Qwen 8B)": "Qwen-8B",
    "SOTA (GPT-5.1)": "GPT-5.1",
    "FrugalGPT (Cascade)": "FrugalGPT",
    "Router (Hybrid)": "Router",
    "Router (Full)": "Router",
    "Router (No RAG)": "Router-NoRAG",
    "Router (Random)": "Router-Random"
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("stats")

def load_latest_data():
    # Search in potential directories
    search_dirs = [INPUT_DIR, "thesis_results_ablation"]
    latest_file = None
    
    for d in search_dirs:
        files = glob.glob(f"{d}/*data_*.csv")
        if files:
            current_latest = max(files, key=os.path.getctime)
            if latest_file is None or os.path.getctime(current_latest) > os.path.getctime(latest_file):
                latest_file = current_latest
    
    if not latest_file:
        logger.error(f"❌ No CSV files found in {INPUT_DIR}/.")
        sys.exit(1)
    
    logger.info(f"📂 Loading data from: {latest_file}")
    df = pd.read_csv(latest_file)
    
    # Normalize columns
    if "judge_score" in df.columns and "quality" not in df.columns:
        df.rename(columns={"judge_score": "quality"}, inplace=True)
    
    # Apply Name Mapping
    df["mode_clean"] = df["mode"].apply(lambda x: NAME_MAP.get(x, x))
    
    # Ensure ID exists for paired testing
    if 'id' not in df.columns:
        logger.warning("⚠️ Column 'id' missing. Generating sequential IDs based on run/mode structure...")
        num_modes = df['mode'].nunique()
        # Assuming data is ordered by Run -> Task -> Mode
        df['id'] = df.groupby('run_id').cumcount() // num_modes

    # Aggregation (Mean across Runs per Question)
    # We average the results of the 5 runs for the same question ID
    agg_dict = {
        "quality": "mean",
        "cost": "mean",
        "latency": "mean"
    }
    if "is_correct" in df.columns:
        agg_dict["is_correct"] = "mean" # Becomes accuracy/probability (0.0 to 1.0)

    df_grouped = df.groupby(["id", "mode_clean"]).agg(agg_dict).reset_index()
    
    return df_grouped

def calculate_cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pool_std = np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)
    if pool_std == 0: return 0.0
    return (np.mean(x) - np.mean(y)) / pool_std

def format_p_value(p):
    if p < 0.001: return "< 0.001"
    return f"= {p:.4f}"

def analyze_metric(df, metric_col, metric_name_en):
    if metric_col not in df.columns:
        return None

    # Pivot: Index=QuestionID, Columns=Models, Values=Metric
    pivot = df.pivot(index="id", columns="mode_clean", values=metric_col).dropna()
    
    # Identify Target Column (Router)
    router_col = next((c for c in pivot.columns if TARGET_MODEL_KEY in c and "No" not in c), None)
    
    if not router_col:
        logger.error(f"❌ Target model containing '{TARGET_MODEL_KEY}' not found in pivot columns: {pivot.columns}")
        return None

    baselines = [c for c in pivot.columns if c != router_col]
    
    # 1. Normality Test (Shapiro-Wilk)
    try:
        _, p_shapiro = stats.shapiro(pivot[router_col])
        is_normal = p_shapiro > ALPHA
    except: is_normal = False

    # 2. Global Difference Test (Friedman)
    try:
        all_groups = [pivot[c] for c in pivot.columns]
        stat_fried, p_fried = stats.friedmanchisquare(*all_groups)
        is_global_sig = p_fried < ALPHA
    except ValueError:
        p_fried, is_global_sig = 1.0, False

    # 3. Post-hoc Tests (Wilcoxon Signed-Rank)
    comparisons = {}
    for base in baselines:
        try:
            # Paired test
            stat_w, p_w = stats.wilcoxon(pivot[router_col], pivot[base])
            d = calculate_cohens_d(pivot[router_col], pivot[base])
            diff_mean = (pivot[router_col] - pivot[base]).mean()
            sig = p_w < ALPHA
        except ValueError:
            # Handles identical data arrays
            p_w, d, diff_mean, sig = 1.0, 0.0, 0.0, False
        
        comparisons[base] = {
            "p_val": p_w, "stat": stat_w, "cohen_d": d,
            "diff_mean": diff_mean, "significant": sig
        }

    return {
        "metric_name": metric_name_en,
        "target": router_col,
        "shapiro_p": p_shapiro if 'p_shapiro' in locals() else 0,
        "is_normal": is_normal,
        "friedman_stat": stat_fried if 'stat_fried' in locals() else 0,
        "friedman_p": p_fried,
        "is_global_sig": is_global_sig,
        "comparisons": comparisons,
        "baselines": baselines
    }

def generate_latex_text(results):
    if not results: return "% Metric not found\n"
    
    m = results["metric_name"]
    target = results["target"]
    
    # Introduction Paragraph
    dist_text = "a normal distribution" if results["is_normal"] else "a non-normal distribution"
    fried_res = "statistically significant differences" if results["is_global_sig"] else "no significant differences"
    
    text = f"""
% -------------------------------------------------------
% Statistical Analysis: {m}
% -------------------------------------------------------
The analysis of \\textbf{{{m}}} compared the proposed \\textit{{{target}}} against multiple baselines. 
The Shapiro-Wilk test indicated {dist_text} for the Router data ($p {format_p_value(results['shapiro_p'])}$).
The Friedman test confirmed {fried_res} among the groups ($\chi^2 = {results['friedman_stat']:.2f}, p {format_p_value(results['friedman_p'])}$).

Post-hoc paired Wilcoxon signed-rank tests revealed:
\\begin{{itemize}}
"""
    
    table_rows = []
    # Sort order: Local -> Frugal -> SOTA -> Ablations
    def sort_key(name):
        if "Gemma" in name or "Qwen" in name: return 0
        if "Frugal" in name: return 1
        if "GPT" in name or "SOTA" in name: return 2
        return 3
        
    sorted_baselines = sorted(results["baselines"], key=sort_key)

    for base in sorted_baselines:
        comp = results["comparisons"][base]
        
        if not comp["significant"]:
            interp = "statistically equivalent"
        else:
            # Determine directionality
            is_higher = comp["diff_mean"] > 0
            
            # For Quality/Accuracy: Higher is Better
            if "Quality" in m or "Accuracy" in m:
                interp = "superior" if is_higher else "inferior"
            # For Cost/Latency: Higher is Worse
            else:
                interp = "less efficient (higher)" if is_higher else "more efficient (lower)"

        text += f"    \\item \\textbf{{{target} vs. {base}}}: The Router proved to be \\textbf{{{interp}}} ($Z = {comp['stat']:.1f}, p {format_p_value(comp['p_val'])}, d={comp['cohen_d']:.2f}$).\n"
        
        sig_mark = '*' if comp['significant'] else 'ns'
        table_rows.append([
            f"{target} vs. {base}",
            f"{comp['diff_mean']:.4f}",
            f"{format_p_value(comp['p_val'])}",
            f"{comp['cohen_d']:.2f}",
            sig_mark
        ])

    text += "\\end{itemize}\n"
    
    # LaTeX Table
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
    print("="*80)
    print("🚀 THESIS STATISTICAL GENERATOR (Multi-Model + FrugalGPT)")
    print("="*80)
    
    df = load_latest_data()
    print(f"ℹ️  Models found: {df['mode_clean'].unique()}")
    
    sections = []
    
    # 1. Objective Accuracy (Ground Truth) - Most Important
    res_acc = analyze_metric(df, "is_correct", "Objective Accuracy")
    if res_acc: sections.append(generate_latex_text(res_acc))
    
    # 2. Subjective Quality (Judge)
    res_qual = analyze_metric(df, "quality", "Judge Quality Score")
    if res_qual: sections.append(generate_latex_text(res_qual))
    
    # 3. Cost
    res_cost = analyze_metric(df, "cost", "Inference Cost")
    if res_cost: sections.append(generate_latex_text(res_cost))
    
    # 4. Latency
    res_lat = analyze_metric(df, "latency", "Latency")
    if res_lat: sections.append(generate_latex_text(res_lat))

    # Save to file
    output_file = f"{INPUT_DIR}/statistical_report_final.tex"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("% Generated by statistical_validation.py\n")
        for sec in sections:
            f.write(sec + "\n")

    print(f"\n✅ LaTeX file generated: {output_file}")
    print("="*80)

if __name__ == "__main__":
    main()