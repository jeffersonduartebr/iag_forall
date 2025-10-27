# 🧭 Roadmap Técnico — 12 Meses

Este roadmap descreve as principais etapas de evolução técnica do **roteador de LLMs multiobjetivo**, que integra múltiplos modelos (locais e comerciais), sistema de *bandits adaptativos*, observabilidade em tempo real e mecanismos de avaliação automática via “juízes” LLM.

---

## 🚀 Visão Geral

O objetivo central do projeto é desenvolver uma arquitetura **modular, autoavaliativa e expansível**, capaz de:
- Orquestrar chamadas a múltiplos modelos LLM (Ollama, Gemini, OpenAI, etc.);
- Avaliar respostas com juízes dinâmicos e multiheurísticos;
- Otimizar a seleção de modelos com base em custo, qualidade e latência (NSGA-II);
- Garantir rastreabilidade, observabilidade e transparência via métricas e dashboards.

---

## 📅 Cronograma de Desenvolvimento (12 Meses)

| Mês | Foco Principal | Entregas Técnicas |
|-----|----------------|-------------------|
| **1–2** | **Fundação da arquitetura** | - Estrutura modular (`app/`, `routers/`, `services/`, `providers/`)<br>- Integração básica com `LiteLLM` e `Ollama`<br>- Configuração de `.env`, `settings.py` e logs estruturados<br>- Criação dos endpoints iniciais (`/query`, `/health`) |
| **3–4** | **Sistema de decisão (Bandits)** | - Implementação de `epsilon-greedy` com persistência (`history.json`)<br>- Salvamento incremental e carregamento automático<br>- Testes de exploração/exploração em cenário multi-modelo<br>- Integração com Prometheus para métricas básicas |
| **5–6** | **Camada de juízes LLM** | - Suporte a múltiplos juízes dinâmicos (`JUDGE_LLMS`)<br>- Julgamento híbrido: heurístico + LLM<br>- Configuração via API para adicionar/remover juízes<br>- Logs e métricas de acurácia dos julgamentos |
| **7–8** | **Algoritmo Multiobjetivo (NSGA-II)** | - Implementação completa do NSGA-II<br>- Integração com camada de decisão (`router_strategy`)<br>- Normalização dos objetivos: custo, latência e qualidade<br>- Visualização das frentes de Pareto em painel de métricas |
| **9–10** | **Observabilidade e dashboards** | - Integração total com **Prometheus** e **Grafana**<br>- Criação de dashboards: performance dos modelos, latência média, distribuição de rewards<br>- Endpoint `/metrics` aprimorado e documentação PromQL |
| **11** | **RAG e contexto adaptativo** | - Suporte opcional a RAG (document retrievers)<br>- Avaliação contextual dos juízes com `use_rag=True`<br>- Métricas de impacto do RAG na qualidade |
| **12** | **Consolidação e automação** | - Testes de carga com **Locust** e benchmarking cruzado<br>- Ajuste fino de hiperparâmetros (ε, α, N dos juízes)<br>- Automação CI/CD via GitHub Actions<br>- Documentação final (API, diagramas, deploy Docker) |

---

## ⚙️ Principais Marcos Técnicos

### 🧩 Módulos Core
- `providers/` — abstração para múltiplos backends (Gemini, Ollama, etc.)
- `judges/` — avaliação de qualidade multi-LLM
- `bandits/` — aprendizado adaptativo (ε-greedy com histórico)
- `router_strategy/` — seleção ótima baseada em NSGA-II
- `settings.py` — parametrização centralizada (pydantic)

### 🧠 Inteligência Adaptativa
- Avaliação contínua de *rewards*
- Atualização automática de pesos de decisão
- Persistência incremental dos resultados
- Suporte a reinicialização e análise retroativa (`history.json`)

### 📊 Observabilidade
- Exposição de métricas Prometheus (`/metrics`)
- Dashboards Grafana: performance por modelo, eficiência dos juízes, Pareto fronts
- Logging estruturado com `uvicorn + logging`

---

## 🧪 Stack Tecnológica

| Componente | Tecnologia |
|-------------|-------------|
| **API / Core** | Python 3.11 + FastAPI |
| **LLM Providers** | LiteLLM + Ollama + Gemini |
| **Decisão / Aprendizado** | ε-Greedy + NSGA-II |
| **Observabilidade** | Prometheus + Grafana |
| **Testes de Carga** | Locust |
| **Persistência** | JSON (histórico local) / Banco opcional |
| **CI/CD** | GitHub Actions + Docker Compose |

---

## 📈 Métricas e Indicadores

- **Qualidade média por modelo**
- **Tempo de resposta (latência)**
- **Custo estimado por 1K tokens**
- **Número de julgamentos automáticos realizados**
- **Distribuição dos rewards (explore/exploit ratio)**
- **Evolução da média ponderada NSGA-II**

---

## 🔮 Extensões Futuras (Ano 2+)

- Armazenamento de histórico em banco relacional (PostgreSQL)
- Interface Web para controle de juízes e modelos
- Integração com frameworks de avaliação como **Helm** e **LangSmith**
- Módulo de aprendizado por reforço contínuo (bandits + PPO)
- Implementação de *reputation weighting* entre juízes

---

## 🧾 Licença e Governança

O projeto segue licença **MIT**, com código aberto e documentação pública.  
Contribuições e *pull requests* são bem-vindos — especialmente nas áreas de:
- Novos provedores LLM;
- Estratégias de seleção multiobjetivo;
- Dashboards de observabilidade.

---

📍 **Última atualização:** Outubro de 2025  
📘 **Responsável técnico:** [Jefferson Duarte (@jeffersonduartebr)](https://github.com/jeffersonduartebr)
