# Trading AI — Setup & Start Guide

## Prerequisites

- Python 3.11 (Windows Store version is fine)
- Node.js 18+ and npm
- Alpaca paper account (free at alpaca.markets)
- Finnhub account (free at finnhub.io)

---

## First-Time Setup

### 1. Fill in API keys

Open `backend/.env` and replace the placeholder values:

```
FINNHUB_API_KEY=your_finnhub_key
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

---

### 2. Create the Python virtual environment

Open a PowerShell terminal in the `backend/` folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, run this first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### 3. Install Python dependencies

With the venv active `(.venv)`:

```powershell
pip install greenlet --only-binary :all:
pip install -r requirements.txt
```

This takes a few minutes. `greenlet` must be installed first from a binary wheel
to avoid a C++ compiler error on Windows.

---

### 4. Install frontend dependencies

Open a second PowerShell terminal in the `frontend/` folder:

```powershell
npm install
```

---

## Every Time You Start the Project

You need **two terminals running at the same time**.

### Terminal 1 — Backend

```powershell
cd backend
.venv\Scripts\Activate.ps1
python -m uvicorn main:app --reload --port 8000
```

Wait for:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Terminal 2 — Frontend

```powershell
cd frontend
npm run dev
```

Then open **http://localhost:5173** in your browser.

---

## Training the Model (run once, in order)

All commands below are run in Terminal 1 (venv active, inside `backend/`).

### Phase 1 — Data & Supervised Training

**Step 1 — Download 5 years of price history** (~2 min):
```powershell
python data/fetch_prices.py
```
Saves Parquet files to `data/cache/`.

**Step 2 — Fetch news and compute sentiment** (needs Finnhub key):
```powershell
python data/fetch_news.py
```

**Step 3 — Build technical indicators and labels**:
```powershell
python data/feature_engineering.py
```
Check output for label distribution (SELL / HOLD / BUY counts).

**Step 4 — Train the TFT model** (interactive menu — pick Full or Light):
```powershell
python train.py
```
A menu appears:
- `[1] Full model` — all 15 tickers, 5 folds, 50 epochs. CPU: ~20-40h, RTX 3060: ~45-90 min.
- `[2] Light (test run)` — 3 tickers, 2 folds, 10 epochs. CPU: ~30-60 min, RTX 3060: ~5-10 min.

Checkpoints saved to `models/checkpoints/`.

**Step 5 — Evaluate accuracy**:
```powershell
python training/evaluate.py
```
Exit criterion: directional accuracy **> 55%** on walk-forward folds.
If it passes, move to Phase 2. If not, retrain with more data or tune hyperparameters in `config.py`.

---

### Phase 2 — Reinforcement Learning Fine-tuning

```powershell
python rl/train_rl.py
```

Takes several hours (1,000,000 PPO timesteps).
Exit criterion: Sharpe **> 1.0** and max drawdown **< 20%** on validation.

---

### Phase 3 — Paper Trading (weeks 11+)

Only start this phase after Phase 2 passes its exit criterion.

1. Start the backend and frontend as described above
2. Go to **http://localhost:5173/control**
3. Press **Start**

The model will run automatically every trading day at **15:30 NY time**:
- Fetches latest prices and news
- Runs FinBERT sentiment
- Computes conviction score (0–10)
- Sends order to Alpaca paper account

The dashboard at **http://localhost:5173** will show live score, positions, and P&L.

Real money should only be considered after **6+ months** of paper trading with:
- Sharpe > 1.0
- Max drawdown < 15%
- Win rate > 53%

---

## Troubleshooting

| Error | Fix |
|---|---|
| `No module named 'X'` | Venv not active — run `.venv\Scripts\Activate.ps1` first |
| `ECONNREFUSED` in browser | Backend not running — start Terminal 1 first |
| `'tuple' object has no attribute 'lower'` | Delete `data/cache/*_prices.parquet` and re-run `fetch_prices.py` |
| `greenlet` build error | Run `pip install greenlet --only-binary :all:` before other installs |
| `model must be a LightningModule` | Use `lightning` package, not `pytorch_lightning` — already fixed in code |
| Port 8000 already in use | Another process is using it — kill it or change port in `main.py` and `vite.config.js` |

---

## Project Structure (quick reference)

```
backend/
  main.py              FastAPI server + WebSocket
  config.py            All settings and hyperparameters
  risk_manager.py      Hard-coded safety rules (never bypass)
  scheduler.py         Daily 15:30 NY job
  alpaca_connector.py  Paper trade execution
  data/                Price download, news sentiment, feature engineering
  models/              TFT model, FinBERT embedder, PPO policy
  rl/                  Trading environment, reward function, PPO training
  training/            Dataset builder, supervised training, evaluation

frontend/
  src/pages/           Dashboard, Metrics, Trades, Control
  src/components/      ScoreGauge, OHLCChart, PortfolioChart, KillSwitch
  src/hooks/           WebSocket auto-reconnect hook
  src/api/             Fetch wrapper for FastAPI
```
