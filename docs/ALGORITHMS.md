# Algorithms Documentation

This document describes the core algorithms used in the Multi-Objective LLM Router system.

## Table of Contents

1. [NSGA-II Multi-Objective Optimization](#nsga-ii-multi-objective-optimization)
2. [Multi-Armed Bandits](#multi-armed-bandits)
3. [Uncertainty Quantification (UQ)](#uncertainty-quantification-uq)
4. [Online Machine Learning](#online-machine-learning)
5. [Quality Judges](#quality-judges)

---

## NSGA-II Multi-Objective Optimization

### Overview

The router uses **NSGA-II (Non-dominated Sorting Genetic Algorithm II)** to balance three competing objectives:

- **Quality**: Response accuracy and relevance (0-10 scale)
- **Latency**: Response time in seconds
- **Cost**: Token usage cost in USD

### Implementation Details

**Location**: `app/nsga_weights_updater.py`

#### Fitness Function

Each model is evaluated using a weighted sum objective:

```
score = w_quality * normalized_quality
      - w_latency * normalized_latency
      - w_cost * normalized_cost
```

Where:
- `w_quality` (default: 1.0): Weight for quality objective
- `w_latency` (default: 0.5): Weight for latency objective
- `w_cost` (default: 100.0): Weight for cost objective

#### Genetic Operators

| Operator | Implementation | Parameters |
|----------|---------------|------------|
| Selection | Tournament Selection | Tournament size: 3 |
| Crossover | Simulated Binary Crossover (SBX) | eta: 20, prob: 0.9 |
| Mutation | Polynomial Mutation | eta: 20, prob: 1/n |

#### Population & Generations

- **Population Size**: 50 individuals
- **Generations**: 40
- **Convergence Check**: Hypervolume indicator stability

#### Update Cycle

The NSGA-II optimizer runs as a background service:

1. Collects recent query logs (configurable lookback window)
2. Computes EMA (Exponential Moving Average) metrics per model
3. Evolves population to find Pareto-optimal weight configurations
4. Updates weights in Redis for real-time routing

---

## Multi-Armed Bandits

### Overview

The router uses a **Meta-Bandit** approach that dynamically selects between different bandit algorithms based on context.

**Location**: `app/bandits.py`

### Available Strategies

#### 1. Epsilon-Greedy

Simple exploration-exploitation tradeoff:

```
with probability ε: explore (random model)
with probability 1-ε: exploit (best known model)
```

- `BANDIT_EPSILON` (default: 0.12): Exploration rate

#### 2. UCB1 (Upper Confidence Bound)

Balances exploration with uncertainty:

```
UCB(a) = Q(a) + c * sqrt(ln(t) / N(a))
```

Where:
- `Q(a)`: Average reward for action a
- `t`: Total number of pulls
- `N(a)`: Number of times action a was selected
- `c`: Exploration constant (default: 1.414)

#### 3. Thompson Sampling

Bayesian approach using Beta distributions:

```
For each model:
  sample θ ~ Beta(α, β)
Select model with highest θ
```

- `α`: Successes + 1
- `β`: Failures + 1

### Context-Aware Selection

The meta-bandit considers:

1. **Query Modality**: text, vision, multimodal
2. **Uncertainty Score**: Novel queries favor exploration
3. **Historical Performance**: EMA-based model statistics

### Reward Computation

```python
reward = (w_quality * quality/10.0
        - w_latency * min(latency, 30)/30.0
        - w_cost * min(cost, 0.1)/0.1) / 3.0
```

Normalized to [0, 1] range.

---

## Uncertainty Quantification (UQ)

### Overview

UQ measures how "novel" a query is compared to previously seen queries, influencing routing decisions.

**Location**: `app/utils/uncertainty.py`

### Calculation Method

1. **Embed Query**: Generate embedding vector using text embedding model
2. **Compute Similarity**: Compare against cluster centroids
3. **Calculate Uncertainty**: Lower similarity = higher uncertainty

```python
similarity = max(cosine_similarity(query_embedding, centroid) for centroid in centroids)
uncertainty = 1.0 - similarity
```

### Impact on Routing

| Uncertainty | Behavior |
|-------------|----------|
| Low (< 0.3) | Prefer local/cheap models (known territory) |
| Medium (0.3-0.7) | Use NSGA-II optimal selection |
| High (> 0.7) | Prefer SOTA models (unknown territory) |

### Threshold Configuration

- `UNCERTAINTY_THRESHOLD` (default: 0.7): High uncertainty threshold
- `UQ_CALIBRATION_ENABLED`: Automatic threshold adjustment

---

## Online Machine Learning

### Overview

The router uses **River** (online ML library) for real-time error prediction.

**Location**: `app/online_predictor.py`

### Model Architecture

- **Algorithm**: Logistic Regression with stochastic gradient descent
- **Features**: Query embedding vector (768 dimensions)
- **Target**: Binary (error / no error)

### Training Loop

```python
# Per query:
1. Predict P(error) using current model
2. If judged: update model with actual outcome
3. Save model state to disk
```

### Integration with Sampling

The predicted error probability influences judge sampling rate:

```python
prob_judge = max(monte_carlo_decay_prob, predicted_error_prob)
```

High error prediction = more likely to trigger quality judge.

### Validation Metrics

- **Brier Score**: Prediction calibration (lower is better)
- **ECE (Expected Calibration Error)**: Reliability of confidence
- `PREDICTOR_BRIER_SCORE_THRESHOLD` (default: 0.25): Acceptable Brier score

---

## Quality Judges

### Overview

LLM-based quality assessment using consensus from multiple judges.

**Location**: `app/judges.py`

### Judge Pipeline

1. **Select Judges**: Choose 1-3 LLMs from configured list
2. **Evaluate**: Each judge scores response (0-10)
3. **Aggregate**: Compute consensus score

### Scoring Criteria

Each judge evaluates:

- **Relevance**: Does the response answer the query?
- **Accuracy**: Is the information correct?
- **Completeness**: Are all aspects addressed?
- **Clarity**: Is the response well-structured?

### Consensus Methods

```python
# Majority voting with outlier rejection
scores = [judge1, judge2, judge3]
if std(scores) > 2.0:
    remove outliers
final_score = mean(remaining_scores)
```

### Sampling Strategy

To reduce cost, judges are sampled probabilistically:

1. **Monte Carlo Decay**: `P = 1 / sqrt(n_samples)`
2. **Error Prediction Boost**: High predicted error increases sampling
3. **SOTA Discount**: GPT-5/Claude responses sampled less (0.1x)
4. **Minimum Floor**: `JUDGE_MIN_SAMPLE_RATE` (default: 5%)

### Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `JUDGES_ENABLED` | 1 | Enable/disable judging |
| `JUDGES_MODE` | llm | Judge type (llm, heuristic) |
| `JUDGES_LOCAL_MODEL` | ollama/phi4:latest | Local judge model |
| `JUDGES_REMOTE_MODEL` | gpt-5-mini | Remote judge model |
| `JUDGES_TIMEOUT_S` | 15 | Judge timeout |

---

## Algorithm Interactions

```
┌─────────────────────────────────────────────────────────────┐
│                      Query Input                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Uncertainty Quantification                  │
│                  (Embedding + Centroid Similarity)           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    NSGA-II Candidate Filter                  │
│              (Hard constraints + Pareto scoring)             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Multi-Armed Bandit                         │
│             (Final model selection with exploration)         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     LLM Inference                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Feedback Loop (Async)                      │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐      │
│  │ Quality Judge │ │ Online ML     │ │ Bandit Update │      │
│  │ (Sampled)     │ │ Update        │ │               │      │
│  └───────────────┘ └───────────────┘ └───────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## Performance Considerations

### Computational Complexity

| Algorithm | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| NSGA-II | O(MN²) per generation | O(N) |
| UCB1 | O(K) per selection | O(K) |
| Thompson Sampling | O(K) per selection | O(K) |
| UQ Centroid Search | O(C * D) | O(C * D) |

Where:
- M = number of objectives
- N = population size
- K = number of models (arms)
- C = number of centroids
- D = embedding dimension

### Caching Strategy

- **EMA Values**: Cached in Redis (60s TTL)
- **Bandit State**: Persisted to Redis after each update
- **Centroids**: Loaded at startup, updated hourly
- **Online ML Models**: Persisted to disk per model

---

## Tuning Guidelines

### Quality-Focused Configuration

```
NSGA_W_QUALITY=2.0
NSGA_W_LATENCY=0.3
NSGA_W_COST=50.0
UNCERTAINTY_THRESHOLD=0.6
```

### Cost-Focused Configuration

```
NSGA_W_QUALITY=0.8
NSGA_W_LATENCY=0.5
NSGA_W_COST=200.0
UNCERTAINTY_THRESHOLD=0.8
```

### Latency-Focused Configuration

```
NSGA_W_QUALITY=0.8
NSGA_W_LATENCY=1.5
NSGA_W_COST=50.0
ADAPTIVE_TIMEOUT_ENABLED=1
```

---

## References

1. Deb, K., et al. (2002). "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II"
2. Auer, P., et al. (2002). "Finite-time Analysis of the Multiarmed Bandit Problem"
3. Thompson, W. R. (1933). "On the Likelihood that One Unknown Probability Exceeds Another"
4. River ML Library: https://riverml.xyz/
