<!-- ========================================================== -->
<!-- 🧠 Projeto: Roteador Multiobjetivo de LLMs - README Oficial -->
<!-- ========================================================== -->

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python 3.11" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-green?logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/LLM-Router-orange?logo=ai" alt="LLM Router" />
  <img src="https://img.shields.io/badge/NSGA--II-Optimizer-purple" alt="NSGA-II" />
  <img src="https://img.shields.io/badge/Prometheus%20%7C%20Grafana-Monitoring-yellow?logo=grafana" alt="Grafana Prometheus" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="MIT License" />
</p>

---

# 🧭 Roteador Multiobjetivo de Modelos de Linguagem (LLMs)

Um sistema inteligente de **orquestração, avaliação e otimização** de modelos de linguagem (LLMs), projetado para equilibrar **custo, qualidade e latência** de forma dinâmica e autônoma.

---

## 🔍 Introdução

### O que é
O **Roteador Multiobjetivo de LLMs** é uma arquitetura modular e extensível que decide, em tempo real, **qual modelo de linguagem** (local ou comercial) deve ser utilizado para cada requisição, com base em múltiplos critérios:  
💰 **Custo operacional**, ⚡ **tempo de resposta (latência)** e 🎯 **qualidade da resposta**.

### O que faz
- Gerencia simultaneamente diversos **provedores de LLMs** (como Ollama, Gemini e OpenAI).  
- Mede e aprende continuamente com o desempenho de cada modelo.  
- Utiliza algoritmos de **otimização multiobjetivo (NSGA-II)** e **bandits adaptativos** para selecionar o modelo ideal.  
- Avalia automaticamente as respostas por meio de **“juízes” LLMs**, que atribuem notas de qualidade com base em heurísticas e comparações.  
- Exibe métricas e logs em **dashboards do Grafana**, com dados coletados via **Prometheus**.

### Como faz
O sistema combina três camadas principais:
1. **Roteamento Inteligente:** Algoritmo NSGA-II que equilibra custo, latência e qualidade.  
2. **Aprendizado Adaptativo:** Bandits e feedback dos juízes para ajustar pesos de decisão.  
3. **Observabilidade:** Métricas de uso, custo e desempenho em tempo real (Prometheus + Grafana).

### Por que faz
Os sistemas de IA generativa têm custos elevados e comportamentos imprevisíveis.  
Este projeto propõe uma alternativa **inteligente e transparente**, que busca:
- **Reduzir custos** operacionais sem perda de qualidade;  
- **Aumentar a confiabilidade** de respostas em pipelines multi-modelo;  
- **Promover autonomia e auditabilidade** em ambientes com múltiplas fontes de IA.  

---

## 📚 Sumário

- [🚀 Visão Geral](#-visão-geral)
- [📅 Cronograma de Desenvolvimento (12 Meses)](#-cronograma-de-desenvolvimento-12-meses)
- [⚙️ Principais Marcos Técnicos](#️-principais-marcos-técnicos)
- [🧪 Stack Tecnológica](#-stack-tecnológica)
- [📈 Métricas e Indicadores](#-métricas-e-indicadores)
- [🔮 Extensões Futuras (Ano 2+)](#-extensões-futuras-ano-2)
- [🎯 Metas de Desenvolvimento (Roadmap Técnico Detalhado)](#-metas-de-desenvolvimento-roadmap-técnico-detalhado)
- [📚 Publicações e Produção Técnica](#-publicações-e-produção-técnica)
- [🌍 Impacto Científico e Tecnológico](#-impacto-científico-e-tecnológico)
- [🧾 Licença e Governança](#-licença-e-governança)

---

## 🚀 Visão Geral

O objetivo central do projeto é desenvolver uma arquitetura **autoavaliativa e expansível**, capaz de:
- Orquestrar múltiplos modelos LLM (Ollama, Gemini, OpenAI, etc.);
- Avaliar respostas com juízes dinâmicos e multiheurísticos;
- Otimizar a escolha de modelos segundo critérios de custo, qualidade e latência;
- Expor dados e métricas em tempo real para auditoria e melhoria contínua.

---

## 📅 Cronograma de Desenvolvimento (12 Meses)

| Mês | Foco Principal | Entregas Técnicas |
|-----|----------------|-------------------|
| **1–2** | **Fundação da arquitetura** | Estrutura modular, integração com LiteLLM/Ollama, logs estruturados, endpoints iniciais. |
| **3–4** | **Sistema de decisão (Bandits)** | Epsilon-greedy com persistência (`history.json`), integração com Prometheus. |
| **5–6** | **Camada de juízes LLM** | Juízes dinâmicos (`JUDGE_LLMS`), API de configuração e avaliação híbrida. |
| **7–8** | **Algoritmo Multiobjetivo (NSGA-II)** | Implementação completa, integração com `router_strategy`, visualização Pareto. |
| **9–10** | **Observabilidade e dashboards** | Prometheus + Grafana integrados, dashboards de eficiência e latência. |
| **11** | **RAG adaptativo** | Recuperação contextual opcional via embeddings, avaliação do impacto. |
| **12** | **Consolidação e automação** | Testes de carga (Locust), CI/CD com GitHub Actions, documentação e release final. |

---

## ⚙️ Principais Marcos Técnicos

- **Core:** `providers/`, `bandits/`, `judges/`, `router_strategy/`, `settings.py`.  
- **IA Adaptativa:** feedback contínuo, média ponderada dinâmica, reexecução retroativa.  
- **Observabilidade:** logs estruturados, métricas Prometheus, dashboards Grafana.

---

## 🧪 Stack Tecnológica

| Componente | Tecnologia |
|-------------|-------------|
| **API / Core** | Python 3.11 + FastAPI |
| **LLM Providers** | LiteLLM + Ollama + Gemini |
| **Decisão / Aprendizado** | ε-Greedy + NSGA-II |
| **Observabilidade** | Prometheus + Grafana |
| **Cache / Fila** | Redis + Celery |
| **Testes de Carga** | Locust |
| **Persistência** | JSON / PostgreSQL (futuro) |
| **CI/CD** | GitHub Actions + Docker Compose |

---

## 📈 Métricas e Indicadores

- Qualidade média por modelo  
- Latência média (p50/p95)  
- Custo por 1K tokens  
- Número de julgamentos automáticos realizados  
- Taxa explore/exploit (Bandit)  
- Evolução da média ponderada NSGA-II  

---

## 🔮 Extensões Futuras (Ano 2+)

- Armazenamento em banco relacional (PostgreSQL)  
- Interface administrativa interativa  
- Integração com frameworks de avaliação (**Helm**, **LangSmith**)  
- Aprendizado por reforço contínuo (Bandits + PPO)  
- *Reputation weighting* entre juízes  

---

## 🎯 Metas de Desenvolvimento (Roadmap Técnico Detalhado)

*(Inclui todas as metas técnicas sobre RAG adaptativo, cache semântico, Celery, observabilidade, juízes adaptativos, etc.)*

---

## 📚 Publicações e Produção Técnica

| Nº | Tipo | Título / Tema | Canal / Periódico-Alvo |
|----|------|----------------|--------------------------|
| **1** | Artigo internacional | *“Multi-Objective Routing of Large Language Models Using NSGA-II and Online Bandits”* | **IEEE Access / Applied Soft Computing (Elsevier)** |
| **2** | Artigo técnico | *“Arquitetura de Juízes LLM: Avaliação de Respostas em Sistemas Multi-Modelo”* | **RBIE / Interacções (UÉvora)** |
| **3** | Artigo científico | *“Roteamento Inteligente de Modelos de Linguagem com Reforço Multiobjetivo”* | **JAIR / ACM TIST** |
| **4** | Short Paper | *“Autoavaliadores LLM e Heurísticas de Consenso: Um Estudo Experimental”* | **SBIE / CONEDU** |
| **5** | Artigo aplicado | *“Plataforma de IA Generativa para Avaliação Educacional: Pipeline Multiobjetivo”* | **Revista Brasileira de Informática na Educação** |
| **6** | Workshop Paper | *“Observabilidade e Rastreabilidade em Pipelines de IA com Prometheus e Grafana”* | **ICMLA Workshop / SBIA Demo Track** |
| **7** | Artigo técnico curto | *“Cache Semântico e RAG Adaptativo em Roteadores de LLMs”* | **arXiv / Medium Tech Blog** |
| **8** | Paper experimental | *“Comparative Analysis of Adaptive Bandits and NSGA-II for LLM Routing”* | **NeurIPS Workshop on Responsible AI Systems** |

### 📘 Outras Ações de Disseminação
- **Tutorial técnico:** Série de artigos detalhando integração NSGA-II + Bandits + FastAPI + LiteLLM  
- **Demo interativo:** Painel de monitoramento ativo com Prometheus e Grafana  
- **Relatório técnico:** Documento público de reprodutibilidade e métricas (Zenodo/DOI)  
- **Submissão institucional:** Relatórios e artigos técnicos para editais de inovação  

> 💡 **Meta científica (Ano 1):** Publicar 8 artigos, 1 tutorial técnico e 1 demonstração pública até o 12º mês.

---

## 🌍 Impacto Científico e Tecnológico

O **Roteador Multiobjetivo de LLMs** contribui diretamente para o avanço da pesquisa aplicada em **IA generativa, otimização multiobjetivo e avaliação automática de sistemas inteligentes**, ao:

1. Propor uma **abordagem inédita de orquestração adaptativa** entre múltiplos LLMs locais e comerciais.  
2. Integrar algoritmos clássicos de otimização evolutiva (NSGA-II) a **mecanismos online de aprendizado por reforço**.  
3. Criar uma camada de **avaliação explicável e auditável** para outputs de IA, reforçando a confiabilidade e transparência.  
4. Promover **reprodutibilidade científica** com logs estruturados, dashboards e métricas abertas (Prometheus).  
5. Servir como **plataforma experimental** para estudos sobre IA responsável, custo-benefício computacional e decisões multi-critério em sistemas generativos.  

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
