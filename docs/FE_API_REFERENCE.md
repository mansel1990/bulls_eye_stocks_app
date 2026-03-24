# Bulls Eye Stocks — API Reference

> For the Next.js PWA at **https://www.mcubetechstudio.com/**

---

## Base URL

```
https://<your-railway-app>.railway.app
```

> Replace with the actual Railway URL once deployed. The team will share this.

---

## Authentication

All `/api/*` endpoints require an API key.

**For regular requests** — send as a header:
```
x-api-key: <secret>
```

**For SSE (EventSource)** — send as a query param (browsers cannot set headers on EventSource):
```
GET /api/advance/stream?key=<secret>
```

Store the key in your Vercel environment variables:
```
NEXT_PUBLIC_BULLS_EYE_API_URL=https://<railway-url>
BULLS_EYE_API_KEY=<secret>           # server-side only (API routes)
NEXT_PUBLIC_BULLS_EYE_API_KEY=<secret>  # if calling from browser
```

> **Security note:** If you call the API from the browser (client components), the key will be visible in JS. For sensitive ops like `/api/reset`, route them through a Next.js API route so the key stays server-side.

---

## Endpoints

### GET /api/status

Returns the full portfolio snapshot. Use this to render the dashboard.

**Query params:**

| Param | Type | Required | Description |
|---|---|---|---|
| `as_of` | `YYYY-MM-DD` | No | View portfolio state on a specific past date. Omit for current. |

**Request:**
```typescript
const res = await fetch(`${API_URL}/api/status`, {
  headers: { "x-api-key": API_KEY },
});
const data = await res.json();
```

**Response:**
```json
{
  "current_date": "2026-03-24",
  "last_run_date": "2026-03-23",
  "viewed_date": "2026-03-24",
  "sim_end": "2026-03-24",

  "open_positions": [
    {
      "ticker": "RELIANCE",
      "entry_date": "2026-03-20",
      "entry_price": 2450.50,
      "current_price": 2480.75,
      "hold_days": 4,
      "unreal_pnl_pct": 1.24,
      "unreal_pnl_rs": 12400.00,
      "amount_invested": 10000.00,
      "peers": "TCS, INFY, WIPRO",
      "entry_z": -1.85,
      "peer_slope_pct": 0.45
    }
  ],

  "todays_buys": [
    {
      "ticker": "SBIN",
      "entry_price": 540.25,
      "entry_z": -1.92,
      "peer_slope_pct": 1.23,
      "peers": "ICICIBANK, AXISBANK"
    }
  ],

  "todays_sells": [
    {
      "ticker": "HDFC",
      "entry_price": 2100.00,
      "exit_price": 2125.50,
      "pnl_pct": 1.21,
      "hold_days": 8,
      "exit_reason": "MEAN"
    }
  ],

  "history": [
    {
      "ticker": "MARUTI",
      "entry_date": "2026-02-15",
      "exit_date": "2026-02-28",
      "entry_price": 8900.00,
      "exit_price": 8750.00,
      "pnl_pct": -1.69,
      "hold_days": 13,
      "amount_invested": 10000.00,
      "status": "SOLD"
    }
  ],

  "metrics": {
    "total_invested": 150000.00,
    "realised_pnl": 2450.75,
    "open_invested": 50000.00,
    "win_rate": 65.3,
    "wins": 49,
    "losses": 26,
    "total_closed": 75,
    "gain_pct": 1.63
  },

  "chart": {
    "dates": ["2026-01-01", "2026-01-02", "..."],
    "open_positions": [0, 1, 3, "..."],
    "bought": [0, 1, 2, "..."],
    "sold": [0, 0, 1, "..."],
    "total_invested": [10000, 20000, "..."],
    "realised_pnl": [0, 100, 250, "..."]
  }
}
```

**Field reference:**

| Field | Description |
|---|---|
| `exit_reason` | `"MEAN"` = z-score returned to 0, `"MAX_HOLD"` = held 30 days |
| `unreal_pnl_pct` | Unrealised P&L as % |
| `unreal_pnl_rs` | Unrealised P&L in Rupees (based on ₹10,000 position) |
| `entry_z` | Z-score at entry (negative = oversold vs peers) |
| `peer_slope_pct` | % slope of peer prices over 30 days (positive = peers trending up) |

---

### GET /api/advance/stream — SSE (Recommended)

Advances the simulation by one trading day. Streams progress in real-time via Server-Sent Events. Use this for the best UX with a progress indicator.

**SSE events:**

```
data: {"pct": 5,   "step": "Connecting to DB...",     "elapsed": 0.12}
data: {"pct": 20,  "step": "Loading tickers...",       "elapsed": 1.8}
data: {"pct": 60,  "step": "Computing z-scores...",    "elapsed": 6.4}
data: {"pct": 100, "step": "Complete",                 "elapsed": 12.4,
       "advanced_to": "2026-03-24",
       "buys_today": 3,
       "sells_today": 2,
       "buy_tickers": ["SBIN", "INFY", "WIPRO"],
       "sell_details": [
         {"ticker": "HDFC", "exit_price": 2125.50, "pnl_pct": 1.21,
          "hold_days": 8, "exit_reason": "MEAN"}
       ]}
```

On error:
```
data: {"pct": 0, "step": "Error: <message>", "elapsed": 0.5}
```

**Next.js usage:**
```typescript
function advanceDay(apiUrl: string, apiKey: string) {
  const url = `${apiUrl}/api/advance/stream?key=${apiKey}`;
  const es = new EventSource(url);

  es.onmessage = (e) => {
    const event = JSON.parse(e.data);

    // Update progress bar
    setProgress(event.pct);
    setStep(event.step);

    if (event.pct === 100) {
      es.close();

      // Trigger notification
      if (event.buys_today > 0 || event.sells_today > 0) {
        new Notification("Bulls Eye Update", {
          body: `${event.buys_today} buys, ${event.sells_today} sells today`,
          icon: "/icon-192.png",
        });
      }

      // Refresh dashboard
      refetchStatus();
    }
  };

  es.onerror = () => {
    es.close();
    setError("Stream failed");
  };
}
```

---

### POST /api/advance — Non-streaming

Same as above but waits for full completion before responding. Use this from server-side Next.js code or if you don't need progress.

**Request:**
```typescript
const res = await fetch(`${API_URL}/api/advance`, {
  method: "POST",
  headers: { "x-api-key": API_KEY },
});
const data = await res.json();
```

**Response:**
```json
{
  "advanced_to": "2026-03-24",
  "buys_today": 3,
  "sells_today": 2,
  "buy_tickers": ["SBIN", "INFY", "WIPRO"],
  "sell_details": [
    {
      "ticker": "HDFC",
      "exit_price": 2125.50,
      "pnl_pct": 1.21,
      "hold_days": 8,
      "exit_reason": "MEAN"
    }
  ]
}
```

> **Note:** This can take 10–60 seconds depending on server load. Prefer `/api/advance/stream` for user-facing flows.

---

### POST /api/reset

Resets the simulation back to `2026-01-01`. **Deletes all positions and trade history.** Route this through a Next.js API route — do not call directly from the browser.

**Request:**
```typescript
const res = await fetch(`${API_URL}/api/reset`, {
  method: "POST",
  headers: { "x-api-key": API_KEY },
});
```

**Response:**
```json
{ "status": "reset", "current_date": "2026-01-01" }
```

---

### GET /api/warmup

Pre-loads all 347 ticker CSVs and caches into server memory. Call this once after the server cold-starts to make the first `/api/advance` fast.

**Request:**
```typescript
await fetch(`${API_URL}/api/warmup`, {
  headers: { "x-api-key": API_KEY },
});
```

**Response:**
```json
{ "loaded_tickers": 347 }
```

---

## Push Notifications (PWA)

The backend does **not** send push notifications. Trigger them from the Next.js PWA after an advance completes.

### Step 1 — Request permission on first load

```typescript
// In your app layout or a useEffect
useEffect(() => {
  if ("Notification" in window) {
    Notification.requestPermission();
  }
}, []);
```

### Step 2 — Fire notification after advance

```typescript
function notifyUser(buys: number, sells: number, buyTickers: string[]) {
  if (Notification.permission !== "granted") return;

  const body = [
    buys > 0 ? `${buys} new position${buys > 1 ? "s" : ""}: ${buyTickers.join(", ")}` : "",
    sells > 0 ? `${sells} position${sells > 1 ? "s" : ""} closed` : "",
  ]
    .filter(Boolean)
    .join(" | ");

  new Notification("Bulls Eye — Today's Signals", {
    body,
    icon: "/icon-192.png",
    badge: "/badge-72.png",
  });
}
```

### Step 3 — For background push (optional, advanced)

If you want push notifications when the PWA is closed, you need:
1. `web-push` npm package in your Next.js app
2. A Next.js API route (`/api/push/subscribe`) that stores push subscriptions
3. A Next.js API route (`/api/push/send`) that sends push after advancing
4. VAPID keys generated via `npx web-push generate-vapid-keys`

This is optional — in-app notifications (Step 2) work fine when the PWA is open.

---

## Vercel Environment Variables

Add these to your Vercel project settings:

```
NEXT_PUBLIC_BULLS_EYE_API_URL=https://<railway-url>.railway.app
NEXT_PUBLIC_BULLS_EYE_API_KEY=<api-key>
```

> Once the Railway URL is live, the team will share the values to fill in.

---

## Error Responses

All errors follow this shape:

```json
{ "error": "Unauthorized" }          // 401 — missing or wrong API key
{ "detail": "Not found" }            // 404 — FastAPI default
{ "detail": "Internal server error"} // 500
```

---

## Data Notes

- All prices are in **Indian Rupees (INR)**
- All stocks are **NSE-listed**
- Position size is fixed at **₹10,000 per trade**
- Simulation runs from **2026-01-01** to today
- `pnl_pct` values are percentages (e.g. `1.21` = 1.21%)
