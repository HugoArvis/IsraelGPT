# CLAUDE.md — Trading AI Project

## Agent Guidelines (Original Rules — Never Override)

- Always write robust and scalable code
- Keep code simple and avoid unnecessary complexity
- Use clear and consistent naming conventions
- Comment briefly in English when necessary (no over-commenting)
- Follow best practices for the chosen language
- Always test code after writing or modifying it
- Verify that changes do not break existing functionality
- If something is uncertain, explicitly mention it
- Log recurring or important mistakes in CLAUDE.log
- Avoid repeating the same mistakes
- NEVER delete or overwrite important files without explicit permission
- Ask questions if any requirement is unclear
- Do NOT make assumptions or invent missing details
- Do not hallucinate APIs, libraries, or features
- Do not generate untested or speculative code

---

## Project Overview

A hybrid deep learning system that predicts stock market signals and executes
paper trades automatically. Combines historical price data (TFT model) and
financial news sentiment (FinBERT) to output a **conviction score from 0 to 10**,
which maps to a position size. A Reinforcement Learning layer (PPO) then
fine-tunes the strategy through simulation before any real capital is used.

**Key output formula:**
```
probs  = model(state)           # [P_sell, P_hold, P_buy] via softmax
score  = 5 + (P_buy - P_sell) * 5   # float in [0.0, 10.0]
position_pct = (score - 5) / 5 * 100  # % of portfolio to hold
```

**Score interpretation:**
```
0.0 – 2.0  →  Strong sell   (position: -100% to -60%)
2.0 – 3.5  →  Partial sell  (position: -60%  to -30%)
3.5 – 6.5  →  Hold          (position: ~0%, minor adjustments)
6.5 – 8.0  →  Partial buy   (position: +30%  to +60%)
8.0 – 10.0 →  Strong buy    (position: +60%  to +100%)
```

**Confidence filter:** if `max(P_sell, P_hold, P_buy) < 0.55` → force score=5.0, no trade.

---

## Project Structure

```
trading-dashboard/
├── backend/                    # All Python — FastAPI + ML + trading logic
│   ├── main.py                 # FastAPI entry point, REST routes, WebSocket
│   ├── scheduler.py            # APScheduler — daily decision at 15:30 NY
│   ├── risk_manager.py         # Hard-coded safety rules (never bypass)
│   ├── alpaca_connector.py     # Alpaca paper trading API wrapper
│   ├── ws_manager.py           # Active WebSocket connection manager
│   ├── config.py               # All constants and hyperparameters
│   ├── .env                    # API keys — NEVER commit this file
│   ├── requirements.txt
│   ├── data/
│   │   ├── fetch_prices.py     # yfinance → local Parquet cache
│   │   ├── fetch_news.py       # Finnhub company news → daily sentiment
│   │   └── feature_engineering.py  # RSI, MACD, Bollinger, ATR, sentiment agg
│   ├── models/
│   │   ├── tft_model.py        # Temporal Fusion Transformer (pytorch-forecasting)
│   │   ├── finbert_embedder.py # FinBERT wrapper (HuggingFace local inference)
│   │   └── custom_policy.py    # TFT as PPO feature extractor (SB3 CustomPolicy)
│   ├── rl/
│   │   ├── trading_env.py      # gym.Env — the trading simulation environment
│   │   ├── reward_functions.py # Incremental Sharpe + drawdown penalty + tx cost
│   │   ├── train_rl.py         # PPO training loop (Stable-Baselines3)
│   │   └── callbacks.py        # EvalCallback + early stopping
│   └── training/
│       ├── dataset.py          # TimeSeriesDataSet (pytorch-forecasting)
│       ├── train_supervised.py # Phase 1 supervised training loop
│       └── evaluate.py         # Walk-forward validation, metrics
│
├── frontend/                   # React + Vite + TailwindCSS
│   ├── src/
│   │   ├── main.jsx
│   │   ├── App.jsx
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx   # Live score gauge + OHLC chart + P&L
│   │   │   ├── Metrics.jsx     # Sharpe, drawdown, win rate charts
│   │   │   ├── Trades.jsx      # Trade log + softmax probabilities
│   │   │   └── Control.jsx     # Start/Stop/Kill switch + risk config
│   │   ├── components/
│   │   │   ├── ScoreGauge.jsx      # Animated 0–10 circular gauge (WebSocket)
│   │   │   ├── OHLCChart.jsx       # TradingView Lightweight Charts
│   │   │   ├── PortfolioChart.jsx  # P&L line chart (recharts)
│   │   │   ├── DrawdownChart.jsx   # Red area drawdown chart (recharts)
│   │   │   └── KillSwitch.jsx      # Red button with confirmation dialog
│   │   ├── hooks/
│   │   │   └── useWebSocket.js     # Custom hook — WS connection + auto-reconnect
│   │   └── api/
│   │       └── client.js           # fetch wrapper → FastAPI :8000
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
│
├── CLAUDE.md                   # This file — read at every session
├── CLAUDE.log                  # Error log — append only, never delete
└── .gitignore                  # Must include: .env, node_modules, __pycache__, *.parquet
```

---

## Tech Stack (100% Free, Verified May 2026)

### Backend — Python
| Library | Version | Purpose |
|---|---|---|
| `fastapi` | >=0.110.0 | REST API + WebSocket server |
| `uvicorn` | >=0.27.0 | ASGI server |
| `apscheduler` | >=3.10.0 | Daily 15:30 NY scheduler |
| `yfinance` | >=0.2.40 | Historical OHLCV (free, no key) |
| `finnhub-python` | >=2.4.20 | News + sentiment + real-time quotes |
| `alpaca-py` | >=0.26.0 | Paper trading execution |
| `ta` | >=0.10.2 | Technical indicators (RSI, MACD…) |
| `transformers` + `torch` | >=4.38 / 2.2 | FinBERT local inference |
| `pytorch-forecasting` | >=1.1.0 | TFT implementation |
| `pytorch-lightning` | >=2.2.0 | Training loop |
| `stable-baselines3` | >=2.3.0 | PPO agent |
| `gymnasium` | >=0.29.1 | Gym environment interface |
| `optuna` | >=3.5.0 | Hyperparameter tuning |
| `vectorbt` | >=0.26.2 | Backtesting |
| `mlflow` | >=2.10.0 | Experiment tracking (local) |
| `python-dotenv` | >=1.0.0 | .env file loading |
| `loguru` | >=0.7.0 | Structured logging |

### Frontend — JavaScript
| Library | Purpose |
|---|---|
| React 18 + Vite 5 | Framework + bundler |
| `react-router-dom` v6 | Client-side routing |
| `lightweight-charts` v4 | TradingView OHLC charts (candlesticks) |
| `recharts` v2 | Sharpe, drawdown, P&L charts |
| `tailwindcss` v3 | Utility-first styling |
| Native `WebSocket` API | Real-time updates (no extra lib needed) |

### APIs (all free tiers)
| API | Use | Limit | Key required |
|---|---|---|---|
| `yfinance` | Historical prices for training | None known | No |
| Finnhub | News, sentiment, real-time quotes | 60 req/min | Yes (free signup) |
| Finnhub WebSocket | Live price stream | 50 symbols max | Yes |
| Alpaca paper | Order execution, paper account | Free forever | Yes (free signup) |

**DO NOT USE:** IEX Cloud (shut down Aug 2024), Polygon.io free tier (too limited),
NewsAPI (not financial-specific enough).

---

## Environment Variables (.env — never commit)

```
FINNHUB_API_KEY=your_key_here
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

Load with: `from dotenv import load_dotenv` then `os.getenv("FINNHUB_API_KEY")`.

---

## Development Phases (build in this exact order)

### Phase 1 — Supervised Pretraining (weeks 1–5)
Goal: TFT + FinBERT learn to classify each trading day as SELL / HOLD / BUY.

1. `fetch_prices.py` — download 5+ years OHLCV for 10–20 US tickers via yfinance,
   save as Parquet locally (never re-download, work from cache).
2. `fetch_news.py` — Finnhub `company_news()` per ticker, run through FinBERT,
   store daily aggregated sentiment scores.
3. `feature_engineering.py` — compute RSI(14), MACD, Bollinger Bands, ATR(14),
   SMA(20/50), EMA(12/26), volume ratio, VIX, day-of-week, quarter-end flag.
4. `dataset.py` — build `TimeSeriesDataSet` with 60-day encoder window,
   5-day prediction horizon.
5. `train_supervised.py` — cross-entropy loss, class weights {SELL:3, HOLD:1, BUY:3}
   to handle class imbalance. Walk-forward validation (never random split).
6. Labelling rule: `relative_return = ticker_return(t→t+5) - SP500_return(t→t+5)`
   SELL if < -1.5%, BUY if > +1.5%, HOLD otherwise.

**Exit criterion:** directional accuracy > 55% on out-of-sample walk-forward folds.

### Phase 2 — Reinforcement Learning Fine-tuning (weeks 6–10)
Goal: PPO agent maximizes risk-adjusted returns using the pretrained TFT.

1. `trading_env.py` — implement `gymnasium.Env`:
   - `observation_space`: normalized 60-day feature window + current position + drawdown
   - `action_space`: `Box(low=0.0, high=10.0, shape=(1,))` — the conviction score
   - `step(action)`: apply position, compute reward, advance one day
   - `reset()`: sample a random 1-year historical episode

2. `reward_functions.py` — reward at each step:
   ```python
   r = sharpe_incremental - 2.0 * drawdown_penalty - 0.5 * transaction_cost
   # sharpe_incremental = return_t / rolling_std(returns, 20)
   # drawdown_penalty   = max(0, current_drawdown - 0.15)
   # transaction_cost   = abs(delta_position) * 0.001
   # terminal bonus     = sharpe_ratio_episode * 10
   ```

3. `custom_policy.py` — freeze pretrained TFT weights, attach MLP head for PPO
   (Option A). Only fine-tune the MLP head initially.

4. `train_rl.py` — PPO hyperparams:
   `lr=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99,
   gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, total_timesteps=1_000_000`

**Exit criterion:** Sharpe > 1.0 and max drawdown < 20% on out-of-sample validation.

### Phase 3 — Paper Trading (weeks 11–36 minimum)
Goal: validate model behavior on real market data with virtual money.

- Backend runs `scheduler.py` daily at **15:30 NY time**
- Workflow: 09:30 fetch news → FinBERT → 15:00 compute features → 15:30 inference
  → compute target position → compare to current position → send order to Alpaca
- All orders go to `https://paper-api.alpaca.markets` (virtual money only)
- Log every decision: score, probabilities, position delta, P&L, Sharpe

**Exit criterion (before real money):** Sharpe > 1.0, max drawdown < 15%,
win rate > 53%, over at least 6 months of paper trading.

---

## FastAPI Routes & WebSocket Contract

### REST Endpoints (backend/main.py)
```
GET  /api/status         → {"model_status": "active"|"stopped", "uptime": "..."}
GET  /api/portfolio       → current positions + total P&L
GET  /api/trades          → list of historical trades with scores + probabilities
GET  /api/metrics         → Sharpe, Sortino, drawdown, win_rate, n_trades
POST /api/model/start     → start the daily scheduler
POST /api/model/stop      → graceful stop (no open position changes)
POST /api/model/kill      → EMERGENCY: liquidate all + stop (requires confirm=true body)
```

### WebSocket (ws://localhost:8000/ws/live)
Push every 1 second during market hours, every 60 seconds otherwise:
```json
{
  "type": "live_update",
  "timestamp": "2026-05-05T15:30:00Z",
  "ticker": "AAPL",
  "score": 7.4,
  "p_sell": 0.08,
  "p_hold": 0.24,
  "p_buy": 0.68,
  "position_pct": 48,
  "portfolio_value": 103420.50,
  "pnl_today_pct": 1.2,
  "pnl_total_pct": 3.4,
  "sharpe_rolling": 1.24,
  "drawdown_current": -2.1,
  "model_status": "active"
}
```

---

## Hard-Coded Risk Rules (risk_manager.py — never bypass)

These rules run **before** any order is sent to Alpaca, regardless of model output:

1. **Global stop-loss:** if portfolio down > 20% from start → liquidate all, stop model
2. **Max position per ticker:** never > 20% of portfolio in one stock
3. **Confidence filter:** if `max(P_sell, P_hold, P_buy) < 0.55` → score = 5.0, no trade
4. **Liquidity filter:** only trade stocks with avg daily volume > 1M shares
5. **Earnings blackout:** freeze all positions 24h before quarterly earnings release
6. **Kill switch:** POST /api/model/kill with `{"confirm": true}` → liquidate + stop

---

## Model Output — Score Conversion Reference

```python
# From softmax probabilities to score and position
p_sell, p_hold, p_buy = model.predict(state)  # sum to 1.0
score = 5.0 + (p_buy - p_sell) * 5.0          # range [0.0, 10.0]
position_pct = (score - 5.0) / 5.0 * 100      # range [-100%, +100%]

# Confidence gate
confidence = max(p_sell, p_hold, p_buy)
if confidence < 0.55:
    score = 5.0
    position_pct = 0.0
```

---

## Local Dev — Start Commands

```bash
# Backend (Terminal 1)
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (Terminal 2)
cd frontend
npm install
npm run dev
# Open http://localhost:5173

# Build for single-process local use
cd frontend && npm run build
# FastAPI serves dist/ on :8000
```

---

## Key Decisions & Constraints Log

| Decision | Reason |
|---|---|
| Classification (3 classes) over regression | MSE on raw prices is unstable; cross-entropy converges reliably |
| Score 0–10 derived from softmax, not predicted directly | Model expresses uncertainty naturally via probabilities |
| Walk-forward validation only | Random split causes data leakage on time series |
| PPO over SAC/TD3 for RL | More stable on noisy financial data, no replay buffer needed |
| TFT pretrained before RL | RL from scratch = blind exploration for weeks |
| FastAPI + React over Streamlit | Real-time WebSocket + TradingView charts require full JS control |
| yfinance for training data | Free, no key, 20+ years history, save to Parquet once |
| Finnhub for news + live data | 60 req/min free, news indexed by ticker, WS for 50 symbols |
| Alpaca paper for execution | Free paper account, real order types, global signup |
| IEX Cloud → DO NOT USE | Permanently shut down August 2024 |
| Polygon.io free → DO NOT USE | Too rate-limited for this use case |

---

## Data Leakage Rules (critical — always verify)

- All features at time `t` must use data **strictly before** `t`
- News cut-off: use only articles published before **09:30 NY** on day `t`
- Price close of day `t` is only available after **16:00 NY** — never use as input feature
- Labels (5-day forward return) reference `t+5`, so they must be excluded from features
- Validation periods must always be **temporally after** training periods

---

## CLAUDE.log Format

Append errors here. Format: `[DATE] [FILE] [TYPE] — description + fix applied`

Example:
```
[2026-05-05] [trading_env.py] [BUG] — reward NaN when rolling_std=0 on first 20 days.
  Fix: added epsilon=1e-8 denominator in sharpe_incremental calculation.
```
