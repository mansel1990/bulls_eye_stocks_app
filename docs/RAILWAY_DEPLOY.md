# Railway Deployment Guide — Bulls Eye Stocks Backend

This guide deploys the FastAPI backend (`correlation_simulator/`) to Railway so it's accessible from https://www.mcubetechstudio.com/.

---

## Prerequisites

- Railway account at https://railway.app (free tier works)
- Railway CLI installed
- Git repo pushed to GitHub (Railway deploys from git)

---

## Step 1 — Install Railway CLI

```bash
npm install -g @railway/cli
```

Verify:
```bash
railway --version
```

---

## Step 2 — Login to Railway

```bash
railway login
```

This opens a browser for authentication.

---

## Step 3 — Prepare the Repository

Before deploying, make sure these files are committed:

```
correlation_simulator/
├── app.py          ✓ (CORS + auth added)
├── db.py           ✓ (env vars)
├── engine.py       ✓ (env vars for file paths)
├── requirements.txt ✓ (python-dotenv added)
├── Procfile        ✓ (tells Railway how to start)
└── .env            ✗ DO NOT COMMIT — this is git-ignored
```

Also commit the data files that the engine needs:
- `daily_data/` folder (all CSV files) — **required**
- `new_technique/stratergies/models/corr_ml_v4b.pkl` — **required**
- `new_technique/stratergies/models/corr_ml_v4b_features.pkl` — **required**
- `new_technique/experiments/sheets/fundamental_score_2026-02-19.xlsx` — **required**

> If these are currently git-ignored (check `.gitignore`), you need to either unignore them or use Railway's volume storage. The simplest approach is to commit them.

Push everything to GitHub:
```bash
git add .
git commit -m "Add Railway deployment config"
git push origin main
```

---

## Step 4 — Create Railway Project

Go to https://railway.app/dashboard → **New Project** → **Deploy from GitHub repo** → select your repo.

**OR** use the CLI:

```bash
# Run from the repo root
railway init
```

When asked for the project name, use something like `bulls-eye-stocks-api`.

---

## Step 5 — Set the Root Directory

Railway needs to know the app lives in `correlation_simulator/`, not the repo root.

In the Railway dashboard:
1. Open your service → **Settings** tab
2. Under **Source** → set **Root Directory** to `correlation_simulator`
3. Under **Deploy** → set **Start Command** to:
   ```
   uvicorn app:app --host 0.0.0.0 --port $PORT
   ```
   (This overrides the Procfile if needed, but the Procfile should handle it automatically.)

---

## Step 6 — Set Environment Variables

In the Railway dashboard → your service → **Variables** tab, add:

| Variable | Value |
|---|---|
| `DB_HOST` | `ep-quiet-unit-a1p4t66q-pooler.ap-southeast-1.aws.neon.tech` |
| `DB_PORT` | `5432` |
| `DB_NAME` | `neondb` |
| `DB_USER` | `neondb_owner` |
| `DB_PASSWORD` | `npg_6gTyXD1qoRLd` |
| `API_KEY` | Generate a strong random string (see below) |
| `ALLOWED_ORIGIN` | `https://www.mcubetechstudio.com` |
| `DATA_DIR` | `../daily_data` |
| `FUND_FILE` | `../new_technique/experiments/sheets/fundamental_score_2026-02-19.xlsx` |
| `MODEL_PATH` | `../new_technique/stratergies/models/corr_ml_v4b.pkl` |
| `FEAT_PATH` | `../new_technique/stratergies/models/corr_ml_v4b_features.pkl` |

**Generate a strong API key:**
```bash
# On Mac/Linux
openssl rand -hex 32

# On Windows PowerShell
-join ((65..90) + (97..122) + (48..57) | Get-Random -Count 32 | % {[char]$_})
```

Save the generated key — you'll need it in Vercel too.

---

## Step 7 — Deploy

Railway auto-deploys when you push to the connected branch. Trigger a manual deploy:

```bash
railway up
```

Or click **Deploy** in the Railway dashboard.

Watch logs in the dashboard. A successful start looks like:
```
INFO:     Started server process
INFO:     Waiting for application startup.
DB schema ready.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:XXXX
[warmup] Engine ready — 347 tickers loaded, z-scores cached
```

---

## Step 8 — Get the Public URL

In Railway dashboard → your service → **Settings** → **Domains** → click **Generate Domain**.

You'll get a URL like: `https://bulls-eye-stocks-api-production.railway.app`

> Share this URL with the FE team. They'll put it in Vercel env vars as `NEXT_PUBLIC_BULLS_EYE_API_URL`.

---

## Step 9 — Test the Deployment

Replace `<RAILWAY_URL>` and `<API_KEY>` below:

```bash
# Health check — get current portfolio status
curl https://<RAILWAY_URL>/api/status \
  -H "x-api-key: <API_KEY>"

# Warmup (call once after first deploy)
curl https://<RAILWAY_URL>/api/warmup \
  -H "x-api-key: <API_KEY>"

# Test auth rejection
curl https://<RAILWAY_URL>/api/status
# Expected: {"error":"Unauthorized"}

# Test CORS from browser console on mcubetechstudio.com
fetch("https://<RAILWAY_URL>/api/status", {
  headers: { "x-api-key": "<API_KEY>" }
}).then(r => r.json()).then(console.log)
```

---

## Step 10 — Add API Key to Vercel

In Vercel dashboard → your project → **Settings** → **Environment Variables**:

```
NEXT_PUBLIC_BULLS_EYE_API_URL = https://<RAILWAY_URL>
NEXT_PUBLIC_BULLS_EYE_API_KEY = <same API_KEY from Railway>
```

Redeploy the Vercel app after adding these.

---

## Ongoing Operations

### Redeploy after code changes
```bash
git push origin main
# Railway auto-deploys
```

### View live logs
```bash
railway logs
```

### Restart the service
Railway dashboard → your service → **Restart**

### Update an env variable
Railway dashboard → Variables → edit → Railway auto-restarts

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: dotenv` | Check `requirements.txt` has `python-dotenv>=1.0.0` |
| `FileNotFoundError: daily_data` | Data files not committed to git — check git status |
| `connection refused` to Neon | Check `DB_PASSWORD` env var is set correctly in Railway |
| CORS blocked on mcubetechstudio.com | Verify `ALLOWED_ORIGIN` is set to `https://www.mcubetechstudio.com` (no trailing slash) |
| 401 Unauthorized | FE is not sending `x-api-key` header (or `?key=` for SSE) |
| SSE stream drops after 30s | Railway default timeout — should be fine, but contact Railway support if persistent |
| First advance is very slow | Normal — warmup cache is rebuilding. Call `/api/warmup` once after deploy. |

---

## Resource Usage (Free Tier)

Railway free tier includes $5/month credit. This app:
- Uses ~256–512MB RAM (ticker data in memory)
- Low CPU except during advance operations
- Should comfortably fit in free tier for personal use

If the server sleeps (Railway may sleep inactive services), the first request will be slow. Upgrade to a paid plan or set up an uptime monitor (e.g., UptimeRobot pinging `/api/status` every 5 minutes) to keep it awake.
