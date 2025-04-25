# Generative AI Routing Platform

![status-badge](https://img.shields.io/badge/status-alpha-blue)
![license-badge](https://img.shields.io/badge/license-MIT-green)

## Introduction

This repository hosts the **Generative AI Routing Platform (GAIRP)**—an open, cloud‑native backbone that dynamically **routes user requests across heterogeneous Large Language Models (LLMs)** and **spins up specialised agents on‑demand**.  
GAIRP is designed for researchers and builders who need a cost‑aware, observable, and easily extensible environment to orchestrate multiple generative models (local or SaaS) under a single, well‑governed API.

## Methodology

GAIRP follows an **incremental roadmap** that delivers value early while de‑risking complex features:

| Phase  | Key Deliverables | Milestones |
|--------|------------------|------------|
| **MVP** | *Gateway* + *static router* + 1 local model + 1 SaaS model | ⬜ API prototype <br> ⬜ Docker‑compose demo |
| **v0.2** | *Agent Manager* + core plugins (SQL, Web Search) | ⬜ Persona DSL <br> ⬜ Tool execution sandbox |
| **v0.3** | *Bandit/Cost‑aware router* + real‑time cost metrics | ⬜ Token & $ tracking <br> ⬜ Grafana dashboards |
| **v1.0** | Self‑service UI + billing + agent marketplace | ⬜ Stripe integration <br> ⬜ Public beta |

Each phase is implemented as a **thin, replaceable microservice** that communicates through an event bus (NATS/JetStream).  

### Technical Stack (snapshot)
* **API Gateway**: Traefik
* **Service Runtime**: FastAPI (Async) + Ray Serve workers
* **Model Serving**: vLLM/TGI containers and OpenAI‑compatible proxies
* **Vector Store**: Qdrant (Hybrid search)
* **Observability**: OpenTelemetry → Prometheus, Grafana, Tempo

## Expected Results

* **Elastic throughput** — automatic GPU / CPU scaling per workload class.
* **Cost transparency** — live dashboards for tokens and USD burn‑rate per tenant.
* **Pluggable agents** — YAML‑defined personas with fine‑grained tool permissions.
* **Governance & security** — JWT, rate‑limit, and audit‑grade trace logs out‑of‑the‑box.
* **Research velocity** — rapid A/B testing of new models via declarative routing rules.

## Related Publications

> *This section is reserved for future conference papers, journal articles, or technical reports related to GAIRP.*

## Contact

| Role | Name | Organisation | Email |
|------|------|--------------|-------|
| Lead Maintainer | **Jefferson Silva** | IFRN – Campus Ipanguaçu | jefferson.duarte@ifrn.edu.br |

Feel free to open an [issue](https://github.com/your‑org/gairp/issues).

