# -*- coding: utf-8 -*-
"""
audit_anomalies.py — Auditoria de Discrepâncias (Sincronizado)
--------------------------------------------------------------
Usa EXATAMENTE o mesmo regex do benchmark para validar se o erro
foi do modelo ou do parser.
"""

import pandas as pd
import glob
import os
import re

INPUT_DIR = "thesis_results"
SCORE_THRESHOLD = 8.0
DISPLAY_LIMIT = 20

def load_latest_data():
    """Carrega latest data."""
    files = glob.glob(f"{INPUT_DIR}/*data_*.csv")
    if not files:
        print(f"❌ No CSV files found.")
        exit(1)
    return pd.read_csv(max(files, key=os.path.getctime))

def debug_parser(dataset_name, model_output):
    """
    Lógica idêntica ao check_correctness do benchmark_thesis.py
    """
    pred = str(model_output).lower().strip()
    
    if dataset_name in ["MMLU", "ARC-Challenge", "HellaSwag", "TruthfulQA"]:
        # Remove markdown bold (**A**) e parênteses (A)
        clean_pred = re.sub(r"[\*\(\)]", "", pred)
        matches = re.findall(r"(?:answer|option|choice)?\s*([a-d0-9])\b", clean_pred)
        return matches[-1] if matches else "NO MATCH"

    if dataset_name == "GSM8K":
        clean_pred = pred.replace(",", "")
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", clean_pred)
        return nums[-1] if nums else "NO NUMBER"
    
    if dataset_name == "BBH-Date":
        return "SUBSTRING CHECK"

    return "N/A"

def main():
    """Executa main."""
    df = load_latest_data()
    
    if 'is_correct' not in df.columns:
        print("❌ CSV missing 'is_correct'.")
        return

    # Anomalia: Errou (0) mas Juiz deu nota alta
    anomalies = df[(df['is_correct'] == 0) & (df['judge_score'] >= SCORE_THRESHOLD)]
    
    print(f"🔍 ANOMALY REPORT: {len(anomalies)} found.")
    
    for idx, row in anomalies.head(DISPLAY_LIMIT).iterrows():
        ds = row.get('dataset', 'Unknown')
        print("-" * 40)
        print(f"ID: {row.get('id')} | {ds} | {row.get('mode')}")
        print(f"GT: {row['is_correct']} | Judge: {row['judge_score']}")
        print(f"Ref: '{row.get('reference')}'")
        print(f"Ans: '{str(row.get('answer', '')).strip()[:100]}...'")
        
        extracted = debug_parser(ds, row.get('answer', ''))
        print(f"Parser saw: '{extracted}'")
        
        # Diagnóstico
        ref = str(row.get('reference', '')).lower().strip()
        ans = str(row.get('answer', '')).lower().strip()
        
        if str(extracted) == ref:
            print("🚨 PARSER ERROR: Script extracted correctly here but failed in benchmark?")
        elif ref in ans:
            print("⚠️  REGEX FAIL: Correct answer is in text, but regex missed it.")
        else:
            print("🧠 JUDGE HALLUCINATION: Answer is wrong, Judge liked it.")

if __name__ == "__main__":
    main()