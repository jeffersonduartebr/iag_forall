#!/bin/bash


pip3 install datasets tqdm pandas httpx --break-system-packages
python3 benchmark_thesis.py
sleep 5
python3 evaluate_results.py
