"""
db.py — PostgreSQL connection + schema creation for the simulator
"""
import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

DB = dict(
    host     = os.getenv("DB_HOST", "ep-quiet-unit-a1p4t66q-pooler.ap-southeast-1.aws.neon.tech"),
    dbname   = os.getenv("DB_NAME", "neondb"),
    user     = os.getenv("DB_USER", "neondb_owner"),
    password = os.getenv("DB_PASSWORD", ""),
    sslmode  = "require",
    port     = int(os.getenv("DB_PORT", 5432)),
)

SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS sim;

-- Simulation state (one row = current pointer)
CREATE TABLE IF NOT EXISTS sim.state (
    id              INT PRIMARY KEY DEFAULT 1,
    sim_date        DATE NOT NULL DEFAULT '2026-01-01',
    last_run_date   DATE,
    total_invested  NUMERIC(16,2) DEFAULT 0,
    total_value     NUMERIC(16,2) DEFAULT 0
);

-- All generated buy signals (one row per ticker per signal date)
CREATE TABLE IF NOT EXISTS sim.stock_suggestions (
    id              SERIAL PRIMARY KEY,
    signal_date     DATE NOT NULL,
    ticker          TEXT NOT NULL,
    peers           TEXT,
    entry_z         NUMERIC(8,4),
    peer_slope_pct  NUMERIC(8,4),
    entry_date      DATE,          -- day user "enters" = signal_date + 1 open
    entry_price     NUMERIC(12,4),
    status          TEXT DEFAULT 'OPEN',   -- OPEN | SOLD | EXPIRED
    exit_date       DATE,
    exit_price      NUMERIC(12,4),
    pnl_pct         NUMERIC(8,4),
    hold_days       INT DEFAULT 0,
    exit_reason     TEXT,
    amount_invested NUMERIC(12,2) DEFAULT 10000,
    UNIQUE(signal_date, ticker)
);

-- Daily portfolio snapshot for the chart
CREATE TABLE IF NOT EXISTS sim.daily_portfolio (
    snap_date       DATE PRIMARY KEY,
    open_positions  INT  DEFAULT 0,
    bought          INT  DEFAULT 0,
    sold            INT  DEFAULT 0,
    total_invested  NUMERIC(16,2) DEFAULT 0,
    realised_pnl    NUMERIC(16,2) DEFAULT 0,
    unrealised_pnl  NUMERIC(16,2) DEFAULT 0
);

-- Ensure state row exists
INSERT INTO sim.state(id, sim_date) VALUES(1, '2026-01-01')
ON CONFLICT(id) DO NOTHING;
"""


def get_conn():
    return psycopg2.connect(**DB)


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(SCHEMA_SQL)
    conn.commit()
    cur.close()
    conn.close()
    print("DB schema ready.")


if __name__ == "__main__":
    init_db()
