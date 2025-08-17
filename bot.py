from __future__ import annotations
import os, time, signal, csv
from dataclasses import dataclass, field
import os
from typing import Tuple, Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta
from pathlib import Path

import ccxt  # type: ignore
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda *a, **k: None

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

console = Console()

# ==========================
# Config
# ==========================
@dataclass
class Config:
    symbol: str = "BTC/USDT:USDT"
    leverage: int = 5
    notional_usdt: float = 10_000.0
    min_edge_bps: float = 12.0
    take_profit_edge_bps: float = 3.0
    max_basis_bps: float = 10.0
    poll_seconds: int = 20
    dry_run: bool = True
    execution_mode: str = "taker"  # maker | taker | hybrid
    # multipair
    multipair: bool = False
    min_volume_usdt: float = 0.0
    volume_x_notional: float = 1000.0
    depth_within_bps: float = 5.0
    min_depth_multiplier: float = 3.0
    max_pairs_to_show: int = 15
    # hybrid settings
    hybrid_wait_seconds: int = 5
    hybrid_min_fill_ratio: float = 0.7
    hybrid_price_offset_bps: float = 0.8
    # logging
    log_dir: str = "logs"
    log_file: str = "trades.csv"
    # funding tracker
    funding_collect: bool = True
    funding_snapshot_secs: int = 300
    funding_lookback_hours: int = 12
    funding_log_file: str = "funding.csv"
    # allow/deny
    allowlist: List[str] = field(default_factory=list)
    denylist: List[str] = field(default_factory=list)
    # spike / timing filters
    max_hours_left: float = 999.0
    spike_min_edge_bps: float = 0.0
    # === dynamic notional ===
    use_dynamic_notional: bool = False
    notional_pct: float = 0.10
    min_notional_usdt: float = 20.0
    max_notional_usdt: float = 1000.0
    reserve_buffer_pct: float = 0.20

def _to_bool(x: str, default=False) -> bool:
    if x is None:
        return default
    return str(x).strip().lower() in ("1", "true", "yes", "on")

def _parse_list_env(name: str) -> List[str]:
    v = os.getenv(name)
    if not v:
        return []
    parts = [p.strip() for p in v.replace(";", ",").split(",")]
    return [p for p in parts if p]

def load_config() -> Config:
    load_dotenv(override=False)  # không ghi đè env có sẵn (Compose)
    cfg = Config()
    # basic
    cfg.symbol = os.getenv("SYMBOL", cfg.symbol)
    cfg.leverage = int(os.getenv("LEVERAGE", cfg.leverage))
    cfg.notional_usdt = float(os.getenv("NOTIONAL_USDT", cfg.notional_usdt))
    cfg.min_edge_bps = float(os.getenv("MIN_EDGE_BPS", cfg.min_edge_bps))
    cfg.take_profit_edge_bps = float(os.getenv("TAKE_PROFIT_EDGE_BPS", cfg.take_profit_edge_bps))
    cfg.max_basis_bps = float(os.getenv("MAX_BASIS_BPS", cfg.max_basis_bps))
    cfg.poll_seconds = int(os.getenv("POLL_SECONDS", cfg.poll_seconds))
    cfg.dry_run = _to_bool(os.getenv("DRY_RUN"), cfg.dry_run)
    cfg.execution_mode = os.getenv("EXECUTION_MODE", cfg.execution_mode).lower()
    # multipair
    cfg.multipair = _to_bool(os.getenv("MULTIPAIR"), cfg.multipair)
    cfg.min_volume_usdt = float(os.getenv("MIN_VOLUME_USDT", cfg.min_volume_usdt))
    cfg.volume_x_notional = float(os.getenv("VOLUME_X_NOTIONAL", cfg.volume_x_notional))
    cfg.depth_within_bps = float(os.getenv("DEPTH_WITHIN_BPS", cfg.depth_within_bps))
    cfg.min_depth_multiplier = float(os.getenv("MIN_DEPTH_MULTIPLIER", cfg.min_depth_multiplier))
    cfg.max_pairs_to_show = int(os.getenv("MAX_PAIRS_TO_SHOW", cfg.max_pairs_to_show))
    # hybrid
    cfg.hybrid_wait_seconds = int(os.getenv("HYBRID_WAIT_SECONDS", cfg.hybrid_wait_seconds))
    cfg.hybrid_min_fill_ratio = float(os.getenv("HYBRID_MIN_FILL_RATIO", cfg.hybrid_min_fill_ratio))
    cfg.hybrid_price_offset_bps = float(os.getenv("HYBRID_PRICE_OFFSET_BPS", cfg.hybrid_price_offset_bps))
    # timing/spike
    cfg.max_hours_left = float(os.getenv("MAX_HOURS_LEFT", cfg.max_hours_left))
    cfg.spike_min_edge_bps = float(os.getenv("SPIKE_MIN_EDGE_BPS", cfg.spike_min_edge_bps))
    # dynamic notional
    cfg.use_dynamic_notional = _to_bool(os.getenv("USE_DYNAMIC_NOTIONAL"), cfg.use_dynamic_notional)
    cfg.notional_pct = float(os.getenv("NOTIONAL_PCT", cfg.notional_pct))
    cfg.min_notional_usdt = float(os.getenv("MIN_NOTIONAL_USDT", cfg.min_notional_usdt))
    cfg.max_notional_usdt = float(os.getenv("MAX_NOTIONAL_USDT", cfg.max_notional_usdt))
    cfg.reserve_buffer_pct = float(os.getenv("RESERVE_BUFFER_PCT", cfg.reserve_buffer_pct))
    # allow/deny from env (comma/semicolon separated)
    al = _parse_list_env("ALLOWLIST")
    dl = _parse_list_env("DENYLIST")
    if al:
        cfg.allowlist = al
    if dl:
        cfg.denylist = dl
    return cfg

# ==========================
# Env status helper (debug)
# ==========================
def print_env_status():
    """Print a safe summary of required API env variables (without exposing secrets)."""
    b_key = bool(os.getenv("BINANCE_API_KEY"))
    b_sec = bool(os.getenv("BINANCE_SECRET"))
    y_key = bool(os.getenv("BYBIT_API_KEY"))
    y_sec = bool(os.getenv("BYBIT_SECRET"))
    Console().print(
        f"[blue]ENV[/blue] BINANCE_API_KEY={'OK' if b_key else 'MISSING'} | BINANCE_SECRET={'OK' if b_sec else 'MISSING'} | "
        f"BYBIT_API_KEY={'OK' if y_key else 'MISSING'} | BYBIT_SECRET={'OK' if y_sec else 'MISSING'}"
    )

# ==========================
# CSV Logger (orders/decisions)
# ==========================
class TradeLogger:
    FIELDS = [
        "ts_iso", "mode", "action", "symbol", "edge_bps", "need_bps", "hours_left",
        "exA", "sideA", "priceA", "amountA", "orderIdA",
        "exB", "sideB", "priceB", "amountB", "orderIdB",
        "notional_usdt", "basis_bps", "fill_ratio_A", "fill_ratio_B",
        "expected_funding_pnl_usdt", "expected_pnl_usdt"
    ]

    def __init__(self, cfg: Config):
        self.dir = Path(cfg.log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / cfg.log_file
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDS)
                writer.writeheader()

    def log(self, row: Dict[str, Any]):
        row.setdefault("ts_iso", datetime.now(timezone.utc).isoformat())
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writerow({k: row.get(k, "") for k in self.FIELDS})

# ==========================
# Funding Tracker (REAL payments)
# ==========================
class FundingTracker:
    FIELDS = [
        "ts_iso", "symbol", "interval_from_iso", "interval_to_iso",
        "binance_count", "binance_sum_usdt", "bybit_count", "bybit_sum_usdt",
        "net_funding_usdt"
    ]

    def __init__(self, cfg: Config):
        self.dir = Path(cfg.log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / cfg.funding_log_file
        self.snapshot_secs = max(60, int(cfg.funding_snapshot_secs))
        self.lookback_ms = max(1, int(cfg.funding_lookback_hours)) * 3600 * 1000
        self.last_snapshot_ms = 0
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDS)
                writer.writeheader()

    def _fetch_history(self, ex, symbol: str, since_ms: int) -> List[Dict[str, Any]]:
        try:
            data = ex.fetch_funding_history(symbol, since_ms, None) or []
            return data
        except Exception:
            return []

    def _sum_payments(self, rows: List[Dict[str, Any]]) -> Tuple[int, float]:
        c = 0
        s = 0.0
        for r in rows:
            amt = r.get("amount")
            try:
                val = float(amt)
            except Exception:
                val = 0.0
            s += val  # positive = received, negative = paid
            c += 1
        return c, s

    def maybe_snapshot(self, cfg: Config, binance, bybit, symbol: str):
        now_ms = int(time.time() * 1000)
        if now_ms - self.last_snapshot_ms < self.snapshot_secs * 1000:
            return
        self.last_snapshot_ms = now_ms
        since = now_ms - self.lookback_ms
        b_rows = self._fetch_history(binance, symbol, since)
        y_rows = self._fetch_history(bybit,   symbol, since)
        b_cnt, b_sum = self._sum_payments(b_rows)
        y_cnt, y_sum = self._sum_payments(y_rows)
        net = b_sum + y_sum
        row = {
            "ts_iso": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "interval_from_iso": datetime.fromtimestamp(since/1000, tz=timezone.utc).isoformat(),
            "interval_to_iso":   datetime.fromtimestamp(now_ms/1000, tz=timezone.utc).isoformat(),
            "binance_count": b_cnt,
            "binance_sum_usdt": f"{b_sum:.6f}",
            "bybit_count": y_cnt,
            "bybit_sum_usdt": f"{y_sum:.6f}",
            "net_funding_usdt": f"{net:.6f}",
        }
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writerow(row)
        console.print(f"[blue]Funding snapshot:[/blue] net={net:.6f} USDT (BIN {b_sum:.6f}, BYB {y_sum:.6f}) for lookback {cfg.funding_lookback_hours}h")

# ==========================
# Utils & math
# ==========================

def has_creds(ex) -> bool:
    return bool(getattr(ex, "apiKey", None)) and bool(getattr(ex, "secret", None))


def _to_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        return float(x)
    except Exception:
        return default


def _position_side(p, contracts: float):
    side = (p.get("side") or "").lower()
    if side in ("long", "short"):
        return side
    if contracts > 0:
        return "long"
    if contracts < 0:
        return "short"
    return "flat"


def taker_bps(ex, symbol: str) -> float:
    try:
        m = ex.market(symbol)
        f = float(m.get("taker", 0.0005))
    except Exception:
        f = 0.0005
    return f * 1e4


def get_quote_volume_usdt(ex, symbol: str) -> float:
    try:
        t = ex.fetch_ticker(symbol)
        qv = _to_float(t.get("quoteVolume"))
        if qv > 0:
            return qv
        bv = _to_float(t.get("baseVolume"))
        last = _to_float(t.get("last")) or _to_float(t.get("close"))
        return bv * last
    except Exception:
        return 0.0


def orderbook_depth_usdt(ex, symbol: str, bps: float) -> Tuple[float, float]:
    try:
        ob = ex.fetch_order_book(symbol, limit=100)
        bid = ob["bids"][0][0] if ob.get("bids") else 0.0
        ask = ob["asks"][0][0] if ob.get("asks") else 0.0
        if bid <= 0 or ask <= 0:
            return 0.0, 0.0
        mid = (bid + ask) / 2.0
        band = mid * (bps / 1e4)
        min_bid_px = mid - band
        max_ask_px = mid + band
        bid_depth = 0.0
        for px, qty in ob.get("bids", [])[:100]:
            if px >= min_bid_px:
                bid_depth += px * qty
            else:
                break
        ask_depth = 0.0
        for px, qty in ob.get("asks", [])[:100]:
            if px <= max_ask_px:
                ask_depth += px * qty
            else:
                break
        return bid_depth, ask_depth
    except Exception:
        return 0.0, 0.0


def common_linear_usdt_symbols(binance, bybit):
    b_syms = {s for s, m in binance.markets.items() if m.get("linear") and m.get("quote") == "USDT" and m.get("contract")}
    y_syms = {s for s, m in bybit.markets.items() if m.get("linear") and m.get("quote") == "USDT" and m.get("contract")}
    return sorted(list(b_syms & y_syms))

def _fetch_free_usdt(ex) -> float:
    """Lấy free USDT cho tài khoản futures của sàn ex (binanceusdm/bybit)."""
    try:
        bal = ex.fetch_balance()
        # ccxt normalized:
        if 'USDT' in bal and isinstance(bal['USDT'], dict):
            # ưu tiên 'free', fallback 'total' - 'used'
            free = _to_float(bal['USDT'].get('free'))
            if free == 0.0:
                total = _to_float(bal['USDT'].get('total'))
                used  = _to_float(bal['USDT'].get('used'))
                calc = total - used if total > 0 else 0.0
                return max(free, calc)
            return free
        # một số sàn nhét trực tiếp vào bal['free'] dạng số:
        f2 = _to_float(bal.get('free'))
        return f2
    except Exception:
        return 0.0

def _safe_get_free_usdt(ex) -> float:
    try:
        bal = ex.fetch_balance()
        usdt = bal.get("USDT") or {}
        free = usdt.get("free")
        if free is None:
            total, used = usdt.get("total", 0.0), usdt.get("used", 0.0)
            free = max(0.0, float(total) - float(used))
        return float(free or 0.0)
    except Exception:
        return 0.0

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def compute_dynamic_notional_usdt(cfg: Config, binance, bybit) -> float:
    if not cfg.use_dynamic_notional:
        return float(cfg.notional_usdt)
    free_bin = _safe_get_free_usdt(binance)
    free_byb = _safe_get_free_usdt(bybit)
    buf = float(cfg.reserve_buffer_pct)
    safe_bin = max(0.0, free_bin * (1.0 - buf))
    safe_byb = max(0.0, free_byb * (1.0 - buf))
    lev = max(1.0, float(cfg.leverage))
    pct = max(0.0, min(1.0, float(cfg.notional_pct)))
    cap_bin = safe_bin * lev * pct
    cap_byb = safe_byb * lev * pct
    cap = min(cap_bin, cap_byb)
    dyn = _clamp(cap, float(cfg.min_notional_usdt), float(cfg.max_notional_usdt))
    try:
        console.print(f"[dim]Dynamic notional: BIN free≈{free_bin:.2f}, BYB free≈{free_byb:.2f}, dyn≈{dyn:.2f} USDT[/dim]")
    except Exception:
        pass
    return float(dyn)


# ==========================
# Exchanges
# ==========================

def init_exchanges():
    api_b_key = os.getenv("BINANCE_API_KEY")
    api_b_sec = os.getenv("BINANCE_SECRET")
    api_y_key = os.getenv("BYBIT_API_KEY")
    api_y_sec = os.getenv("BYBIT_SECRET")

    binance = ccxt.binanceusdm({
        "apiKey": api_b_key or None,
        "secret": api_b_sec or None,
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"adjustForTimeDifference": True, "recvWindow": 10000},
    })
    binance.set_sandbox_mode(False)
    try:
        binance.load_time_difference()
    except Exception:
        pass
    binance.load_markets()

    bybit = ccxt.bybit({
        "apiKey": api_y_key or None,
        "secret": api_y_sec or None,
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "swap", "recvWindow": 10000},
    })
    bybit.set_sandbox_mode(False)
    bybit.load_markets()

    return binance, bybit

# ==========================
# Funding & positions
# ==========================

def ensure_leverage(ex, symbol: str, lev: int, dry_run: bool):
    if dry_run or not has_creds(ex):
        return
    try:
        ex.setLeverage(lev, symbol)
    except Exception as e:
        console.print(f"[yellow]Leverage set failed on {ex.id}: {e}[/yellow]")


def fetch_position_notional(ex, symbol: str, dry_run: bool = False) -> Tuple[float, float]:
    if dry_run or not has_creds(ex):
        return 0.0, 0.0
    try:
        positions = ex.fetch_positions([symbol])
    except Exception:
        positions = []
    long_n = short_n = 0.0
    for p in positions:
        if p.get("symbol") != symbol:
            continue
        contracts = _to_float(p.get("contracts"))
        notional = _to_float(p.get("notional"))
        if notional == 0.0:
            price = _to_float(p.get("markPrice")) or _to_float(p.get("entryPrice"))
            notional = abs(contracts) * price
        side = _position_side(p, contracts)
        if side == "long" and notional > 0:
            long_n += notional
        elif side == "short" and notional > 0:
            short_n += notional
    return long_n, short_n


def get_basis_bps(price_a: float, price_b: float) -> float:
    mid = (price_a + price_b) / 2.0
    if mid <= 0:
        return 0.0
    return ((price_b - price_a) / mid) * 1e4


def fetch_funding_rate_safe(ex, symbol: str) -> dict:
    try:
        return ex.fetch_funding_rate(symbol) or {}
    except Exception:
        return {}


def _hours_to_next_8h_utc(now_ms: int) -> float:
    now = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
    slots = [0, 8, 16]
    for h in slots:
        cand = now.replace(hour=h, minute=0, second=0, microsecond=0)
        if cand > now:
            return (cand - now).total_seconds() / 3600.0
    next_dt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return (next_dt - now).total_seconds() / 3600.0


def read_next_funding(ex, symbol: str) -> Tuple[float, float]:
    d = fetch_funding_rate_safe(ex, symbol)
    fr_next = _to_float(d.get("nextFundingRate"), 0.0)
    if fr_next == 0.0:
        fr_next = _to_float(d.get("fundingRate"), 0.0)
    hours_left = 0.0
    ts = d.get("nextFundingTimestamp") or d.get("nextFundingTime")
    try:
        now = ex.milliseconds()
        if ts:
            ts_i = int(ts)
            if ts_i < 10**12:
                ts_i *= 1000
            hours_left = max(0.0, (ts_i - now) / 3_600_000.0)
        else:
            hours_left = _hours_to_next_8h_utc(now)
    except Exception:
        hours_left = _hours_to_next_8h_utc(ex.milliseconds())
    return fr_next, hours_left


def funding_edge_now_bps(fr_short: float, fr_long: float, hours_left: float, interval_hours: float = 8.0) -> float:
    frac = max(0.0, min(1.0, hours_left / interval_hours)) if interval_hours > 0 else 0.0
    return (fr_short - fr_long) * 1e4 * frac

# ==========================
# Expected PnL calculation
# ==========================

def expected_funding_usdt(edge_bps: float, notional: float) -> float:
    return (edge_bps / 1e4) * notional

# ==========================
# Execution helpers (maker/taker/hybrid) + logging
# ==========================

def _maker_price_from_orderbook(ex, symbol: str, side: str, offset_bps: float) -> Optional[float]:
    try:
        ob = ex.fetch_order_book(symbol, limit=5)
        best_bid = ob["bids"][0][0] if ob.get("bids") else 0.0
        best_ask = ob["asks"][0][0] if ob.get("asks") else 0.0
        if best_bid <= 0 or best_ask <= 0:
            return None
        if side == "sell":
            return best_ask * (1 + offset_bps / 1e4)
        else:
            return best_bid * (1 - offset_bps / 1e4)
    except Exception:
        return None


def _apply_amount_limits(ex, symbol: str, amount: float, price: Optional[float]) -> float:
    try:
        m = ex.market(symbol)
        # precision (số chữ số thập phân)
        prec = None
        if isinstance(m.get("precision"), dict):
            prec = m["precision"].get("amount")
        if prec is not None and prec >= 0:
            amount = round(float(amount), int(prec))
        # min amount
        limits = m.get("limits") or {}
        min_amt = (limits.get("amount") or {}).get("min")
        if min_amt is not None and amount < float(min_amt):
            amount = float(min_amt)
        # min cost (theo giá nếu có)
        min_cost = (limits.get("cost") or {}).get("min")
        if price and min_cost is not None:
            if amount * float(price) < float(min_cost):
                amount = float(min_cost) / float(price)
        return float(amount)
    except Exception:
        return float(amount)


def _qty_from_notional(ex, symbol: str, notional_usdt: float, price: float) -> float:
    if price <= 0:
        return 0.0
    amt = notional_usdt / price
    amt = _apply_amount_limits(ex, symbol, amt, price)
    return amt


def _safe_create_order(ex, symbol: str, type_: str, side: str, amount: float, price: Optional[float] = None, params: Optional[Dict[str, Any]] = None):
    params = params or {}
    return ex.create_order(symbol, type_, side, amount, price, params)


def _order_filled_ratio(ex, order) -> float:
    try:
        oid = order.get("id")
        if not oid:
            return 0.0
        fresh = ex.fetch_order(oid, order.get("symbol"))
        filled = _to_float(fresh.get("filled"))
        amount = _to_float(fresh.get("amount"))
        return (filled / amount) if amount > 0 else 0.0
    except Exception:
        return 0.0


def _cancel_silent(ex, order):
    try:
        oid = order.get("id")
        if oid:
            ex.cancel_order(oid, order.get("symbol"))
    except Exception:
        pass


def place_delta_neutral(bin_short: bool, notional_usdt: float, symbol: str, binance, bybit, dry_run: bool, mode: str, cfg: Config, logger: TradeLogger, meta: Dict[str, Any]):
    sideA, sideB = ("sell", "buy") if bin_short else ("buy", "sell")
    exA, exB = (binance, bybit) if bin_short else (bybit, binance)

    # prices for qty
    tA = exA.fetch_ticker(symbol)
    tB = exB.fetch_ticker(symbol)
    pA = _to_float(tA.get("last")) or _to_float(tA.get("mark")) or _to_float(tA.get("close"))
    pB = _to_float(tB.get("last")) or _to_float(tB.get("mark")) or _to_float(tB.get("close"))

    log_base = dict(
        mode=mode,
        action="decision",
        symbol=symbol,
        edge_bps=f"{meta.get('edge_best', 0.0):.6f}",
        need_bps=f"{meta.get('need', 0.0):.6f}",
        hours_left=f"{meta.get('hours_left', 0.0):.3f}",
        exA=exA.id, sideA=sideA, priceA=f"{pA:.8f}", amountA="",
        orderIdA="", exB=exB.id, sideB=sideB, priceB=f"{pB:.8f}", amountB="", orderIdB="",
        notional_usdt=f"{notional_usdt:.2f}", basis_bps=f"{meta.get('basis_bps',0.0):.3f}",
        fill_ratio_A="", fill_ratio_B="",
        expected_funding_pnl_usdt=f"{expected_funding_usdt(meta.get('edge_best',0.0), notional_usdt):.4f}",
        expected_pnl_usdt=f"{expected_funding_usdt(meta.get('edge_best',0.0), notional_usdt):.4f}"
    )

    if dry_run:
        console.print(f"[cyan]DRY-RUN: {exA.id} {sideA} / {exB.id} {sideB} | mode={mode} | notional≈{notional_usdt:.0f}[/cyan]")
        log_base.update(action="dry_run")
        logger.log(log_base)
        return

    try:
        if mode == "taker":
            amtA = _qty_from_notional(exA, symbol, notional_usdt, pA)
            amtB = _qty_from_notional(exB, symbol, notional_usdt, pB)
            oA = _safe_create_order(exA, symbol, "market", sideA, amtA)
            oB = _safe_create_order(exB, symbol, "market", sideB, amtB)
            log_base.update(action="placed_taker", amountA=f"{amtA}", amountB=f"{amtB}", orderIdA=oA.get("id",""), orderIdB=oB.get("id",""))
            logger.log(log_base)
            console.print("[green]Placed taker orders on both legs[/green]")
            return

        # maker or hybrid -> post-only
        offset = cfg.hybrid_price_offset_bps
        priceA = _maker_price_from_orderbook(exA, symbol, sideA, offset)
        priceB = _maker_price_from_orderbook(exB, symbol, sideB, offset)
        if not priceA or not priceB:
            console.print("[red]Cannot fetch orderbook for maker pricing[/red]")
            if mode == "hybrid":
                return place_delta_neutral(bin_short, notional_usdt, symbol, binance, bybit, dry_run, "taker", cfg, logger, meta)
            return
        amtA = _qty_from_notional(exA, symbol, notional_usdt, priceA)
        amtB = _qty_from_notional(exB, symbol, notional_usdt, priceB)
        paramsA = {"postOnly": True}
        paramsB = {"postOnly": True}
        if getattr(exA, "id", "") == "binanceusdm":
            paramsA["timeInForce"] = "GTX"
        if getattr(exB, "id", "") == "binanceusdm":
            paramsB["timeInForce"] = "GTX"
        oA = _safe_create_order(exA, symbol, "limit", sideA, amtA, priceA, paramsA)
        oB = _safe_create_order(exB, symbol, "limit", sideB, amtB, priceB, paramsB)
        logger.log({**log_base, "action": "placed_maker", "amountA": amtA, "amountB": amtB, "priceA": f"{priceA:.8f}", "priceB": f"{priceB:.8f}", "orderIdA": oA.get("id",""), "orderIdB": oB.get("id","")})
        console.print("[green]Placed maker (post-only) orders[/green]")

        if mode == "maker":
            return

        # HYBRID wait and evaluate
        time.sleep(max(1, int(cfg.hybrid_wait_seconds)))
        rA = _order_filled_ratio(exA, oA)
        rB = _order_filled_ratio(exB, oB)
        logger.log({**log_base, "action": "hybrid_check", "fill_ratio_A": f"{rA:.4f}", "fill_ratio_B": f"{rB:.4f}"})
        if rA >= cfg.hybrid_min_fill_ratio and rB >= cfg.hybrid_min_fill_ratio:
            console.print("[green]Hybrid maker filled sufficiently. Keeping maker orders.[/green]")
            logger.log({**log_base, "action": "hybrid_keep_maker", "fill_ratio_A": f"{rA:.4f}", "fill_ratio_B": f"{rB:.4f}"})
            return
        # cancel + finish with taker for remaining
        _cancel_silent(exA, oA)
        _cancel_silent(exB, oB)
        remA = max(0.0, 1.0 - rA) * notional_usdt
        remB = max(0.0, 1.0 - rB) * notional_usdt
        rem = max(remA, remB)
        if rem < 1:
            logger.log({**log_base, "action": "hybrid_all_filled"})
            return
        tA2 = exA.fetch_ticker(symbol)
        tB2 = exB.fetch_ticker(symbol)
        pA2 = _to_float(tA2.get("last")) or pA
        pB2 = _to_float(tB2.get("last")) or pB
        amtA2 = _qty_from_notional(exA, symbol, rem, pA2)
        amtB2 = _qty_from_notional(exB, symbol, rem, pB2)
        oA2 = _safe_create_order(exA, symbol, "market", sideA, amtA2)
        oB2 = _safe_create_order(exB, symbol, "market", sideB, amtB2)
        logger.log({**log_base, "action": "hybrid_fallback_taker", "amountA": amtA2, "amountB": amtB2, "priceA": f"{pA2:.8f}", "priceB": f"{pB2:.8f}", "orderIdA": oA2.get("id",""), "orderIdB": oB2.get("id","")})
        console.print("[green]Hybrid taker completion done[/green]")

    except Exception as e:
        console.print(f"[red]Order placement failed: {e}[/red]")
        logger.log({**log_base, "action": "error", "exA": exA.id, "exB": exB.id})

# ==========================
# Loops (single/multipair)
# ==========================

def single_pair_loop(binance, bybit, cfg: Config, logger: TradeLogger, funder: Optional[FundingTracker]):
    symbol = cfg.symbol
    ensure_leverage(binance, symbol, cfg.leverage, cfg.dry_run)
    ensure_leverage(bybit, symbol, cfg.leverage, cfg.dry_run)

    extra = []
    if cfg.max_hours_left < 999:
        extra.append(f"maxH={cfg.max_hours_left}h")
    if cfg.spike_min_edge_bps > 0:
        extra.append(f"spike>={cfg.spike_min_edge_bps}bps")
    extra_str = (" | " + " ".join(extra)) if extra else ""

    header = f"Cross-Exchange Funding Bot | symbol={symbol} | lev={cfg.leverage}x | notional={cfg.notional_usdt:.1f} USDT | dry_run={cfg.dry_run} | mode={cfg.execution_mode}{extra_str}"
    running = True
    def handle_sigint(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, handle_sigint)

    while running:
        console.print(Panel.fit(header, style="bold white on blue"))
        # prices
        t_bin = binance.fetch_ticker(symbol)
        t_byb = bybit.fetch_ticker(symbol)
        px_bin = _to_float(t_bin.get("last")) or _to_float(t_bin.get("mark")) or _to_float(t_bin.get("close"))
        px_byb = _to_float(t_byb.get("last")) or _to_float(t_byb.get("mark")) or _to_float(t_byb.get("close"))
        basis_bps = get_basis_bps(px_bin, px_byb)
        # funding & time
        fr_bin_next, hleft_bin = read_next_funding(binance, symbol)
        fr_byb_next,  hleft_byb = read_next_funding(bybit,   symbol)
        if hleft_bin > 0 and hleft_byb > 0:
            hours_left = min(hleft_bin, hleft_byb)
        elif hleft_bin > 0 or hleft_byb > 0:
            hours_left = max(hleft_bin, hleft_byb)
        else:
            hours_left = _hours_to_next_8h_utc(binance.milliseconds())
        edge_A = funding_edge_now_bps(fr_bin_next, fr_byb_next, hours_left)
        edge_B = funding_edge_now_bps(fr_byb_next, fr_bin_next, hours_left)
        best_is_bin_short = edge_A >= edge_B
        best_edge = edge_A if best_is_bin_short else edge_B
        # fees baseline
        entry_fee_bps = 0.0 if cfg.execution_mode in ("maker","hybrid") else (taker_bps(binance, symbol) + taker_bps(bybit, symbol))
        min_edge_effective = max(cfg.min_edge_bps, entry_fee_bps)

        # Timing + spike filters
        if abs(basis_bps) > cfg.max_basis_bps:
            console.print(f"[yellow]Skip: basis {basis_bps:.2f} > max_basis_bps {cfg.max_basis_bps}[/yellow]")
        elif cfg.max_hours_left < 999 and hours_left > cfg.max_hours_left:
            console.print(f"[yellow]Skip: hours_left {hours_left:.2f} > max_hours_left {cfg.max_hours_left}[/yellow]")
        elif cfg.spike_min_edge_bps > 0 and abs(best_edge) < cfg.spike_min_edge_bps:
            console.print(f"[yellow]Skip: edge {best_edge:.3f} < spike_min_edge_bps {cfg.spike_min_edge_bps:.3f}[/yellow]")
        elif best_edge >= min_edge_effective:
            dyn_notional = compute_dynamic_notional_usdt(cfg, binance, bybit)
            meta = {"edge_best": best_edge, "need": min_edge_effective, "hours_left": hours_left, "basis_bps": basis_bps}
            place_delta_neutral(best_is_bin_short, dyn_notional, symbol, binance, bybit, cfg.dry_run, cfg.execution_mode, cfg, logger, meta)
        else:
            console.print(f"[yellow]No entry: edge {best_edge:.3f} < need {min_edge_effective:.3f} bps[/yellow]")

        # Funding snapshot (REAL)
        if funder:
            funder.maybe_snapshot(cfg, binance, bybit, symbol)

        time.sleep(max(1, int(cfg.poll_seconds)))


def multipair_scan_and_trade(binance, bybit, cfg: Config, logger: TradeLogger, funder: Optional[FundingTracker]):
    extra = []
    if cfg.max_hours_left < 999:
        extra.append(f"maxH={cfg.max_hours_left}h")
    if cfg.spike_min_edge_bps > 0:
        extra.append(f"spike>={cfg.spike_min_edge_bps}bps")
    extra_str = (" | " + " ".join(extra)) if extra else ""

    header = f"Cross-Exchange Funding Bot | lev={cfg.leverage}x | notional={cfg.notional_usdt:.1f} USDT | dry_run={cfg.dry_run} | mode={cfg.execution_mode} | multipair=ON{extra_str}"
    running = True
    def handle_sigint(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, handle_sigint)

    while running:
        console.print(Panel.fit(header, style="bold white on blue"))
        syms = common_linear_usdt_symbols(binance, bybit)
        console.print(f"[blue]Found {len(syms)} common symbols[/blue]")  # Add debug
        
        rows = []
        dyn_min_vol = max(cfg.min_volume_usdt, cfg.volume_x_notional * cfg.notional_usdt)
        min_depth_usdt = cfg.min_depth_multiplier * cfg.notional_usdt
        console.print(
            f"     Multipair scan (minVol=max({cfg.min_volume_usdt:,.0f}, "
            f"{cfg.volume_x_notional}×{cfg.notional_usdt:,.0f})={dyn_min_vol:,.0f} USDT, "
            f"depth≥{min_depth_usdt:,.0f} within ±{cfg.depth_within_bps:.1f} bps)"
        )
        
        filtered_count = 0
        for sym in syms:
            # apply allow/deny filter
            if cfg.allowlist and sym not in cfg.allowlist:
                continue
            if cfg.denylist and sym in cfg.denylist:
                continue
            try:
                v_bin = get_quote_volume_usdt(binance, sym)
                v_byb = get_quote_volume_usdt(bybit, sym)
                if v_bin < dyn_min_vol or v_byb < dyn_min_vol:
                    continue
                    
                filtered_count += 1
                if filtered_count <= 5:  # Log first 5 symbols for debug
                    console.print(f"[cyan]Processing {sym}: vol_bin={v_bin/1e6:.1f}M, vol_byb={v_byb/1e6:.1f}M[/cyan]")
                
                t_bin = binance.fetch_ticker(sym)
                t_byb = bybit.fetch_ticker(sym)
                p_bin = _to_float(t_bin.get("last")) or _to_float(t_bin.get("mark")) or _to_float(t_bin.get("close"))
                p_byb = _to_float(t_byb.get("last")) or _to_float(t_byb.get("mark")) or _to_float(t_byb.get("close"))
                basis = get_basis_bps(p_bin, p_byb)
                if abs(basis) > cfg.max_basis_bps:
                    continue
                bid_b, ask_b = orderbook_depth_usdt(binance, sym, cfg.depth_within_bps)
                bid_y, ask_y = orderbook_depth_usdt(bybit, sym, cfg.depth_within_bps)
                if min(bid_b, ask_b, bid_y, ask_y) < min_depth_usdt:
                    continue
                fr_bin, h_bin = read_next_funding(binance, sym)
                fr_byb, h_byb = read_next_funding(bybit, sym)
                if h_bin > 0 and h_byb > 0:
                    h = min(h_bin, h_byb)
                elif h_bin > 0 or h_byb > 0:
                    h = max(h_bin, h_byb)
                else:
                    h = _hours_to_next_8h_utc(binance.milliseconds())
                eA = funding_edge_now_bps(fr_bin, fr_byb, h)
                eB = funding_edge_now_bps(fr_byb, fr_bin, h)
                # timing / spike filters
                if cfg.max_hours_left < 999 and h > cfg.max_hours_left:
                    continue
                best_bin_short = eA >= eB
                best_e = eA if best_bin_short else eB
                if cfg.spike_min_edge_bps > 0 and abs(best_e) < cfg.spike_min_edge_bps:
                    continue
                entry_fee = 0.0 if cfg.execution_mode in ("maker","hybrid") else taker_bps(binance, sym) + taker_bps(bybit, sym)
                need = max(cfg.min_edge_bps, entry_fee)
                score = best_e - need
                rows.append({
                    "symbol": sym, "basis": basis, "edge": best_e, "need": need, "score": score,
                    "hours": h, "dir": ("BIN short" if best_bin_short else "BYB short"),
                    "vbin": v_bin, "vbyb": v_byb
                })
            except Exception as e:
                if filtered_count <= 5:
                    console.print(f"[red]Error processing {sym}: {e}[/red]")
                continue

        console.print(f"[blue]Processed {filtered_count} symbols, found {len(rows)} candidates[/blue]")
        rows.sort(key=lambda r: r["score"], reverse=True)
        top = rows[:cfg.max_pairs_to_show]

        t2 = Table(title=f"Multipair scan (minVol≈{dyn_min_vol:,.0f} USDT, depth≥{min_depth_usdt:,.0f} within ±{cfg.depth_within_bps} bps)", box=box.SIMPLE_HEAVY)
        for col in ["#","Symbol","Basis(bps)","Edge(bps)","Need(bps)","Score","H left","Dir","Vol BIN","Vol BYB"]:
            t2.add_column(col, justify="right" if col in {"#","Basis(bps)","Edge(bps)","Need(bps)","Score","H left","Vol BIN","Vol BYB"} else "left")
        for i, r in enumerate(top, 1):
            t2.add_row(str(i), r["symbol"], f"{r['basis']:.2f}", f"{r['edge']:.3f}", f"{r['need']:.3f}", f"{r['score']:.3f}", f"{r['hours']:.2f}", r["dir"], f"{r['vbin']/1e6:.1f}M", f"{r['vbyb']/1e6:.1f}M")
        console.print(t2)

        if top and top[0]["score"] >= 0:
            best = top[0]
            console.print(f"[green]Best candidate meets threshold: {best['symbol']} (score {best['score']:.3f})[/green]")
            meta = {"edge_best": best['edge'], "need": best['need'], "hours_left": best['hours'], "basis_bps": best['basis']}
            # ensure leverage for the chosen symbol
            ensure_leverage(binance, best["symbol"], cfg.leverage, cfg.dry_run)
            ensure_leverage(bybit,   best["symbol"], cfg.leverage, cfg.dry_run)
            # dùng notional động
            dyn_notional = compute_dynamic_notional_usdt(cfg, binance, bybit)
            place_delta_neutral(best["dir"].startswith("BIN"), dyn_notional, best["symbol"], binance, bybit, cfg.dry_run, cfg.execution_mode, cfg, logger, meta)
        else:
            console.print("[yellow]No candidate meets edge threshold[/yellow]")

        # Funding snapshot (REAL) for top symbol (to limit API usage)
        if funder and top:
            funder.maybe_snapshot(cfg, binance, bybit, top[0]["symbol"])

        time.sleep(max(1, int(cfg.poll_seconds)))

# ==========================
# Entrypoint
# ==========================

def main():
    cfg = load_config()
    print_env_status()
    binance, bybit = init_exchanges()
    logger = TradeLogger(cfg)
    funder = FundingTracker(cfg) if cfg.funding_collect else None

    if cfg.multipair:
        return multipair_scan_and_trade(binance, bybit, cfg, logger, funder)
    else:
        return single_pair_loop(binance, bybit, cfg, logger, funder)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print(f"[red]Fatal error:[/red] {e}")
        raise
