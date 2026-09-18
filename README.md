# 🌱 Darukaa.Earth — Causal Evidence Engine

<p align="center">
  <img src="docs/screenshots/darukaa-dashboard.png" alt="Darukaa.Earth Dashboard" width="100%">
</p>

<p align="center">
  <b>Turn field observations into causal decisions.</b><br>
  An evidence-grounded AI environmental reasoning system.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/PostgreSQL-Database-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/pgvector-Vector_Search-336791" alt="pgvector">
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/LLM-Groq-orange" alt="Groq">
</p>

---

## 📌 Overview

**Darukaa.Earth** is an AI-powered **Causal Evidence Engine (CEE)** for environmental decision support.

Instead of generating generic environmental advice, the system connects:

**Field observations → Environmental variables → Causal relationships → Scientific evidence → Actionable recommendations**

The system combines a structured environmental knowledge graph, scientific evidence retrieval, multi-metric reasoning, and an LLM explanation layer.

---

## 🎯 Problem

Environmental conditions are interconnected.

For example:

```text
Low Soil Carbon
       ↓
Poor Soil Health
       ↓
Reduced Water Retention
       ↓
Biodiversity Impact
```

A useful recommendation therefore needs to reason across multiple environmental variables instead of treating each problem independently.

---

## 🧠 Key Features

- 🌱 **Environmental Site-State Extraction**
  - Converts natural-language field observations into structured environmental variables.

- 🔗 **Causal Knowledge Graph**
  - Represents relationships between interventions, mechanisms, and environmental metrics.

- 📚 **Scientific Evidence Retrieval**
  - Grounds recommendations in research papers, reports, and extracted evidence chunks.

- 🧮 **Multi-Metric Reasoning**
  - Connects variables such as soil carbon, water retention, biodiversity, and pest pressure.

- 💬 **Conversational Interface**
  - Allows users to describe environmental conditions using natural language.

- 🔍 **Evidence & Provenance**
  - Recommendations can be traced back to supporting scientific sources and evidence.

- ⏱️ **Time Horizon & Confidence**
  - Recommendation outputs can include expected time horizon and confidence information.

---

## 🖥️ Interface

The frontend presents the environmental reasoning process through:

- Detected site conditions
- Deficiency analysis
- Environmental metrics
- Causal recommendations
- Evidence and provenance
- Confidence and time horizon

### Dashboard

![Darukaa.Earth Dashboard](docs/screenshots/darukaa-dashboard.png)

---

## 🏗️ Architecture

```text
                         ┌──────────────────┐
                         │       User       │
                         │ Natural Language │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     FastAPI      │
                         │     Backend      │
                         └────────┬─────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │  Site-State Analyzer │
                       │ Environmental Inputs │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │ Causal Reasoning     │
                       │ Engine               │
                       │                      │
                       │ Graph Traversal      │
                       │ Candidate Discovery  │
                       │ Multi-Metric Logic   │
                       └──────────┬───────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
          ┌──────────────────┐        ┌──────────────────┐
          │ PostgreSQL       │        │ Evidence Search  │
          │ + pgvector       │        │ Scientific       │
          │                  │        │ Evidence Chunks  │
          │ Nodes / Edges    │        └────────┬─────────┘
          │ Sources          │                 │
          │ Benchmarks       │                 │
          └────────┬─────────┘                 │
                   └────────────┬──────────────┘
                                ▼
                       ┌──────────────────┐
                       │ Recommendation   │
                       │ Generation       │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Groq LLM         │
                       │ Explanation      │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Final Response   │
                       │                  │
                       │ Recommendation   │
                       │ Reasoning        │
                       │ Metrics          │
                       │ Evidence         │
                       │ Time Horizon     │
                       └──────────────────┘
```

---

## 🗄️ Knowledge Base

The knowledge layer uses **PostgreSQL + pgvector** to combine structured causal relationships with scientific evidence retrieval.

Current knowledge base includes:

| Component | Count |
|---|---:|
| Scientific Sources | 12 |
| Knowledge Nodes | 32 |
| Causal Edges | 63+ |
| Evidence Chunks | 1000+ |
| Soil Carbon Benchmarks | 3 |

> Counts reflect the current project state used for the submission.

---

## 🔬 Example

### User Input

```text
My field has low soil carbon, poor water retention, and low biodiversity.
```

### System Understanding

```text
Soil Organic Carbon       → Low
Water Holding Capacity    → Poor
Associated Biodiversity   → Low
Biome                     → Cropland
```

### Reasoning

```text
Observed Conditions
        ↓
Deficiency Detection
        ↓
Causal Graph Traversal
        ↓
Candidate Interventions
        ↓
Scientific Evidence Retrieval
        ↓
Multi-Metric Reasoning
        ↓
Grounded Recommendation
```

---

## 🧰 Technology Stack

| Layer | Technologies |
|---|---|
| Backend | Python, FastAPI |
| Database | PostgreSQL, pgvector |
| AI / LLM | Groq |
| Reasoning | Causal graph traversal + deterministic analysis |
| Frontend | HTML, CSS, JavaScript |
| Infrastructure | Docker, Docker Compose |
| Version Control | Git, GitHub |

---

## 🔌 API

### Health

```http
GET /health
```

### Environmental Analysis

```http
POST /analyze
```

### Conversational Reasoning

```http
POST /chat
```

---

## 🚀 Local Setup

### Prerequisites

- Python 3.11+
- Docker Desktop
- Git
- uv

### Clone

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd darukaa-cee
```

### Environment Variables

Create `.env`:

```env
DATABASE_URL=postgresql+psycopg://cee:cee@localhost:5432/cee
GROQ_API_KEY=your_groq_api_key
```

**Never commit `.env` or API keys to GitHub.**

### Start

```bash
docker compose up --build
```

API:

```text
http://localhost:8000
```

---


