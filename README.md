# AI-Powered Portfolio Optimizer

A production-grade AI system that combines LLM-based financial news sentiment analysis with quantitative portfolio optimization to generate intelligent stock rebalancing recommendations.

> Built by Parmod | Stack: Python · LangChain · FinBERT · FAISS · FastAPI · Streamlit · Docker

---

## What Problem Does This Solve?

Retail investors make portfolio decisions based on either pure numbers (ignoring news) or pure gut feeling (ignoring math). This tool combines both:

- **Quantitative signal** — Modern Portfolio Theory, Sharpe ratio optimization
- **Sentiment signal** — FinBERT analysis of real-time financial news via RAG pipeline
- **LLM reasoning** — GPT/Gemini explains *why* it recommends each rebalancing action

---

## Architecture

```
User Input (stock tickers + portfolio weights)
                    │
                    ▼
        ┌─────────────────────┐
        │   Streamlit Frontend │  ← frontend/app.py
        └─────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │   FastAPI Backend    │  ← src/api/main.py
        └─────────────────────┘
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
┌──────────────────┐  ┌─────────────────────┐
│  Data Pipeline   │  │  Optimization Layer  │
│  src/data/       │  │  src/optimization/   │
│                  │  │                      │
│ - yfinance       │  │ - Sharpe ratio       │
│ - NewsAPI        │  │ - MPT weights        │
│ - FAISS store    │  │ - Risk calculation   │
└──────────────────┘  └─────────────────────┘
          │
          ▼
┌──────────────────┐
│   LLM Pipeline   │
│   src/models/    │
│                  │
│ - FinBERT        │
│ - LangChain RAG  │
│ - GPT-4 / Gemini │
└──────────────────┘
          │
          ▼
  Recommendation Output
  (weights + sentiment + reasoning)
```

---

## Features

- **Real-time stock data** — fetches price history using yfinance
- **Financial news RAG** — retrieves and embeds recent news per stock using FAISS
- **Sentiment analysis** — FinBERT scores each news article (positive/negative/neutral)
- **Portfolio optimization** — Sharpe ratio maximization using scipy
- **LLM reasoning** — LangChain + GPT-4 explains each recommendation in plain English
- **REST API** — FastAPI backend, fully documented at /docs
- **Interactive UI** — Streamlit dashboard with charts and downloadable report
- **Dockerized** — single command deployment

---

## Project Structure

```
portfolio-optimizer/
│
├── src/
│   ├── data/
│   │   ├── stock_fetcher.py        # yfinance integration
│   │   ├── news_fetcher.py         # NewsAPI integration
│   │   └── vector_store.py         # FAISS embedding store
│   │
│   ├── models/
│   │   ├── sentiment.py            # FinBERT sentiment pipeline
│   │   ├── rag_pipeline.py         # LangChain RAG over news
│   │   └── llm_reasoner.py         # GPT-4/Gemini explanation layer
│   │
│   ├── optimization/
│   │   ├── portfolio.py            # Sharpe ratio, MPT optimization
│   │   └── risk.py                 # Volatility, correlation, VaR
│   │
│   └── api/
│       ├── main.py                 # FastAPI app
│       ├── routes.py               # API endpoints
│       └── schemas.py              # Pydantic models
│
├── frontend/
│   └── app.py                      # Streamlit dashboard
│
├── tests/
│   ├── test_data.py
│   ├── test_sentiment.py
│   ├── test_optimization.py
│   └── test_api.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sentiment_analysis.ipynb
│   └── 03_optimization_experiments.ipynb
│
├── docs/
│   ├── architecture.md
│   ├── api_reference.md
│   └── setup.md
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Quickstart

```bash
# Clone
git clone https://github.com/Parmodk2310/portfolio-optimizer.git
cd portfolio-optimizer

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Install dependencies
pip install -r requirements.txt

# Run backend
uvicorn src.api.main:app --reload

# Run frontend (new terminal)
streamlit run frontend/app.py
```

**Or with Docker:**

```bash
docker-compose up --build
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/optimize` | Submit tickers, get optimized weights |
| GET | `/sentiment/{ticker}` | Get sentiment score for a stock |
| GET | `/news/{ticker}` | Get recent news for a stock |
| GET | `/portfolio/report` | Download full analysis report |
| GET | `/health` | Health check |

Full API docs available at `http://localhost:8000/docs` after running.

---

## Environment Variables

```bash
# .env.example
OPENAI_API_KEY=your_openai_key
NEWS_API_KEY=your_newsapi_key
GOOGLE_API_KEY=your_gemini_key       # optional, if using Gemini
FAISS_INDEX_PATH=./data/faiss_index
LOG_LEVEL=INFO
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Stock Data | yfinance | Free, reliable, no auth needed |
| News Data | NewsAPI | Free tier, 100 requests/day |
| Sentiment | FinBERT | Finance-specific BERT, more accurate than general models |
| Embeddings | sentence-transformers | Fast, runs locally |
| Vector Store | FAISS | Fast similarity search, no infra needed |
| LLM Orchestration | LangChain | Already know it from internship |
| LLM | GPT-4 / Gemini 1.5 Pro | Interchangeable via LangChain |
| Optimization | scipy + numpy | Standard for portfolio math |
| Backend | FastAPI | Async, auto-docs, production-ready |
| Frontend | Streamlit | Fast to build, easy to deploy |
| Containerization | Docker | Reproducible deployment |
| Deployment | Hugging Face Spaces | Free, supports Streamlit |

---

## Build Order (Follow This Exactly)

**Week 1 — Data pipeline first**
- [ ] `stock_fetcher.py` — fetch price history for given tickers
- [ ] `news_fetcher.py` — fetch recent news articles per ticker
- [ ] `vector_store.py` — embed and store news in FAISS
- [ ] Notebook 01: explore the data, make sure it works

**Week 2 — LLM layer**
- [ ] `sentiment.py` — run FinBERT on each article, return score
- [ ] `rag_pipeline.py` — retrieve relevant news chunks per ticker
- [ ] `llm_reasoner.py` — generate plain English recommendation
- [ ] Notebook 02: test sentiment on real articles

**Week 3 — Optimization layer**
- [ ] `portfolio.py` — Sharpe ratio calculation, weight optimization
- [ ] `risk.py` — volatility, correlation matrix, basic VaR
- [ ] Combine sentiment score + optimization output
- [ ] Notebook 03: test on real portfolio (AAPL, MSFT, GOOGL, etc.)

**Week 4 — API + Frontend + Deploy**
- [ ] `main.py` + `routes.py` + `schemas.py` — FastAPI backend
- [ ] `frontend/app.py` — Streamlit dashboard with charts
- [ ] `Dockerfile` + `docker-compose.yml`
- [ ] Deploy on Hugging Face Spaces
- [ ] Write final README with demo GIF

---

## Research References

- [FinBERT: Financial Sentiment Analysis with Pre-trained Language Models](https://arxiv.org/abs/1908.10063)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Modern Portfolio Theory — Markowitz (1952)
- Python for Finance — Yves Hilpisch
- Options, Futures and Other Derivatives — John Hull

---

## License

MIT License — free to use, modify, and distribute.

---

## Author

**Parmod** | ML Engineer  
Email: parmodk.official@gmail.com  
LinkedIn: [linkedin.com/in/parmodk2310](https://www.linkedin.com/in/parmodk2310/)  
GitHub: [github.com/Parmodk2310](https://github.com/Parmodk2310)
