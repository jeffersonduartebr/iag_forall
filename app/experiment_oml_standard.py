# -*- coding: utf-8 -*-
"""
experiment_oml_standard.py — OML Comparison with Statistical Validation
-----------------------------------------------------------------------
Compares Logistic Regression, ARF, and MLP using Distributed k-Fold Cross-Validation.
Includes Friedman and Wilcoxon tests to determine the SOTA approach.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import glob
import os
import concurrent.futures
from tqdm import tqdm
from scipy import stats
import itertools

# River Imports
from river import linear_model, forest, neural_net, optim, preprocessing, compose, metrics, activations
from sentence_transformers import SentenceTransformer

# Configuração
INPUT_DIR = "thesis_results"
OUTPUT_DIR = "paper_results_standard"
K_FOLDS = 10  # Validação Cruzada (Recomendado: 5 ou 10)
MAX_WORKERS = 12  # Ajuste conforme seus núcleos
ALPHA = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 1. DATA LOADING & PREP
# ==============================================================================
def load_data():
    """Carrega data."""
    print("📂 Loading benchmark data...")
    files = glob.glob(f"{INPUT_DIR}/*data_*.csv")
    if not files:
        if os.path.exists(f"{INPUT_DIR}/benchmark_checkpoint.csv"):
            latest = f"{INPUT_DIR}/benchmark_checkpoint.csv"
        else:
            raise FileNotFoundError("No data found. Run benchmark_thesis.py first.")
    else:
        latest = max(files, key=os.path.getctime)
    
    print(f"   File: {latest}")
    df = pd.read_csv(latest)
    
    if 'is_correct' not in df.columns:
        raise ValueError("Column 'is_correct' missing.")
    
    # Limpeza e Ordenação por Dataset (Simular Drift)
    df = df.dropna(subset=['query', 'is_correct'])
    df = df.sort_values(by='dataset')
    
    print("🧠 Generating embeddings (Sentence-BERT)...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    embeddings = encoder.encode(df['query'].tolist(), show_progress_bar=True)
    
    X = [{f"d{i}": v for i, v in enumerate(vec)} for vec in embeddings]
    y = [1 if val == 0 else 0 for val in df['is_correct']] # Target: 1 = Erro
    
    return X, y, df['dataset'].tolist()

# ==============================================================================
# 2. MODEL DEFINITIONS
# ==============================================================================
def get_model(name):
    """Obtém model."""
    if name == "Logistic Regression":
        model = linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.01))
    elif name == "Adaptive Random Forest":
        model = forest.ARFClassifier(n_models=10, seed=42)
    elif name == "Online MLP":
        model = neural_net.MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activations=(activations.ReLU, activations.ReLU, activations.Sigmoid),
            optimizer=optim.Adam(lr=0.001),
            seed=42
        )
    else:
        raise ValueError(f"Unknown model: {name}")

    return compose.Pipeline(preprocessing.StandardScaler(), model)

# ==============================================================================
# 3. WORKER FUNCTION
# ==============================================================================
def run_kfold_for_model(model_name, X, y, k_folds):
    """Executa kfold for model."""
    print(f"   🚀 Worker started for: {model_name}")
    
    ensemble = [get_model(model_name) for _ in range(k_folds)]
    fold_metrics = [metrics.Accuracy() for _ in range(k_folds)]
    
    acc_history = [[] for _ in range(k_folds)]
    latency_history = [] 
    
    for i, (xi, yi) in enumerate(zip(X, y)):
        test_idx = i % k_folds
        
        for k in range(k_folds):
            model = ensemble[k]
            
            if k == test_idx:
                # TESTE
                start = time.perf_counter_ns()
                y_pred = model.predict_one(xi)
                duration = time.perf_counter_ns() - start
                
                fold_metrics[k].update(yi, y_pred)
                acc_history[k].append(fold_metrics[k].get())
                latency_history.append(duration / 1e6) # ms
            else:
                # TREINO
                model.learn_one(xi, yi)
                last = acc_history[k][-1] if acc_history[k] else 0.0
                acc_history[k].append(last)
                
    # Retorna o histórico completo E a acurácia final de cada fold para estatística
    final_accuracies = [m.get() for m in fold_metrics]
    
    return model_name, acc_history, latency_history, final_accuracies

# ==============================================================================
# 4. STATISTICAL ANALYSIS (NOVO)
# ==============================================================================
def analyze_statistics(final_results):
    """
    Realiza testes de Friedman e Wilcoxon para determinar o melhor algoritmo.
    final_results: dict {model_name: [acc_fold1, acc_fold2, ...]}
    """
    print("\n" + "="*80)
    print("📊 STATISTICAL ANALYSIS (Friedman + Wilcoxon)")
    print("="*80)

    models = list(final_results.keys())
    data = [final_results[m] for m in models]
    
    # 1. Tabela Descritiva
    means = [np.mean(d) for d in data]
    stds = [np.std(d) for d in data]
    
    print(f"{'Model':<25} | {'Mean Acc':<10} | {'Std Dev':<10}")
    print("-" * 50)
    for i, m in enumerate(models):
        print(f"{m:<25} | {means[i]:.4f}     | {stds[i]:.4f}")
    print("-" * 50)

    # 2. Teste de Friedman (Global)
    # H0: Todos os algoritmos performam igual
    stat, p_value = stats.friedmanchisquare(*data)
    print(f"\n🔹 Friedman Test: Chi2={stat:.2f}, p-value={p_value:.4e}")
    
    latex_output = ""
    
    if p_value > ALPHA:
        print("   ❌ No statistically significant difference found between algorithms.")
        latex_output += f"The Friedman test indicated no significant differences ($\chi^2={stat:.2f}, p={p_value:.3f}$)."
    else:
        print("   ✅ Significant difference detected! Running Post-hoc (Wilcoxon)...")
        latex_output += f"The Friedman test revealed significant differences ($\chi^2={stat:.2f}, p < 0.001$). Post-hoc analysis (Wilcoxon):"
        
        # 3. Post-hoc: Pairwise Wilcoxon com correção de Bonferroni
        # Compara todos contra todos
        pairs = list(itertools.combinations(range(len(models)), 2))
        p_values = []
        
        print("\n   Post-hoc Pairwise Comparisons:")
        for i, j in pairs:
            stat_w, p_w = stats.wilcoxon(data[i], data[j])
            p_values.append(p_w)
            
        # Correção de Bonferroni (multiplica p pelo número de pares)
        # Ou Holm-Bonferroni (mais poderoso, mas simples serve aqui)
        adj_p_values = [min(1.0, p * len(pairs)) for p in p_values]
        
        best_model_idx = np.argmax(means)
        print(f"   🏆 Best Numerical Mean: {models[best_model_idx]}")
        
        for (i, j), p_adj in zip(pairs, adj_p_values):
            sig = "*" if p_adj < ALPHA else "ns"
            m1, m2 = models[i], models[j]
            print(f"   - {m1} vs {m2}: p_adj={p_adj:.4f} ({sig})")
            
            if p_adj < ALPHA:
                winner = m1 if means[i] > means[j] else m2
                latex_output += f"\n\\\\ \\textbf{{{winner}}} significantly outperformed {m1 if winner==m2 else m2} ($p={p_adj:.3f}$)."

    # Salva LaTeX
    with open(f"{OUTPUT_DIR}/stats_summary.tex", "w") as f:
        f.write(latex_output)
    print(f"\n📄 LaTeX summary saved to {OUTPUT_DIR}/stats_summary.tex")

# ==============================================================================
# 5. MAIN EXECUTION & PLOTTING
# ==============================================================================
def main():
    """Executa main."""
    X, y, datasets = load_data()
    models = ["Logistic Regression", "Adaptive Random Forest", "Online MLP"]
    
    results_acc_hist = {}
    results_lat = {}
    results_final_acc = {} # Para estatística
    
    print(f"\n🧪 Starting Parallel Execution ({MAX_WORKERS} workers, {K_FOLDS}-Fold)...")
    start_total = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(run_kfold_for_model, m, X, y, K_FOLDS): m 
            for m in models
        }
        
        for future in concurrent.futures.as_completed(futures):
            m_name, acc_hist, lat_hist, final_accs = future.result()
            results_acc_hist[m_name] = acc_hist
            results_lat[m_name] = lat_hist
            results_final_acc[m_name] = final_accs
            print(f"   ✅ Finished: {m_name}")

    print(f"⏱️ Total time: {time.time() - start_total:.2f}s")

    # --- STATISTICAL ANALYSIS ---
    analyze_statistics(results_final_acc)

    # --- PLOTTING ---
    sns.set_theme(style="whitegrid")
    
    # Fig 1: Accuracy over Time
    plt.figure(figsize=(12, 6))
    colors = {"Logistic Regression": "blue", "Adaptive Random Forest": "green", "Online MLP": "red"}
    
    for name, history in results_acc_hist.items():
        data = np.array(history).T
        mean = np.mean(data, axis=1)
        std = np.std(data, axis=1)
        x = range(len(mean))
        plt.plot(x, mean, label=name, color=colors[name])
        plt.fill_between(x, mean - std, mean + std, color=colors[name], alpha=0.15)

    boundaries = [i for i in range(1, len(datasets)) if datasets[i] != datasets[i-1]]
    for b in boundaries:
        plt.axvline(x=b, color='gray', linestyle='--', alpha=0.5)
        if b < len(datasets) * 0.95:
            plt.text(b+2, 0.5, "Drift", rotation=90, color='gray', fontsize=8)

    plt.title(f"Online Learning: Adaptation to Concept Drift ({K_FOLDS}-Fold)", fontweight='bold')
    plt.xlabel("Samples Processed")
    plt.ylabel("Prequential Accuracy")
    plt.legend(loc='lower right')
    plt.savefig(f"{OUTPUT_DIR}/fig1_accuracy_drift.png", dpi=300)
    
    # Fig 2: Latency Boxplot
    lat_df = []
    for name, lats in results_lat.items():
        for l in lats: lat_df.append({"Model": name, "Latency (ms)": l})
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=pd.DataFrame(lat_df), x="Model", y="Latency (ms)", showfliers=False, palette="viridis")
    plt.title("Inference Latency Distribution", fontweight='bold')
    plt.savefig(f"{OUTPUT_DIR}/fig2_latency.png", dpi=300)
    
    print(f"\n🏆 Done. Results in {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
