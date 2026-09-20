#!/bin/bash
# Objective: Shell utility for benchmark.


uv pip install --system datasets tqdm pandas httpx
python3 benchmark_thesis.py
sleep 5
python3 evaluate_results.py
