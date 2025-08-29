from __future__ import annotations
import os, time, csv
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List, Set
from datetime import datetime, timezone, timedelta
from pathlib import Path
import threading  # thêm ở đầu file

from dotenv import load_dotenv  # NEW
import ccxt  # type: ignore
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from telegram_manager import TelegramManager  # keep import

console = Console()

# ---- Runtime / Telegram state ----
@dataclass
class BotState:
    started_ts: float = time.time()
    paused: bool = False
    last_loop_ts: float = 0.0
    entry_count: int = 0
    exit_count: int = 0
    last_error: str = ""

TELE: Optional[TelegramManager] = None
STATE = BotState()
RUNNING = True  # NEW: global switch to exit loops

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
    basis_stop_delta_bps: float = 10.0
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
    log_dir: str = "./logs"
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
    # Price arbitrage take profit & stoploss
    price_arbitrage_tp_pct: float = float(os.getenv("PRICE_ARBITRAGE_TP_PCT", 1.5)) 
    price_arbitrage_sl_pct: float = float(os.getenv("PRICE_ARBITRAGE_SL_PCT", -2.0))
    price_arbitrage_min_hold_minutes: int = int(os.getenv("PRICE_ARBITRAGE_MIN_HOLD_MINUTES", 10))
    funding_countdown_minutes: int = int(os.getenv("FUNDING_COUNTDOWN_MINUTES", 30))  

    print(f"[CONFIG] TP={price_arbitrage_tp_pct}%, SL={price_arbitrage_sl_pct}%, "
        f"MIN_HOLD={price_arbitrage_min_hold_minutes}m, FUNDING_COUNTDOWN={funding_countdown_minutes}m")

    # --- HARD PRICE TP ---
    hard_price_tp: bool = bool(int(os.getenv("HARD_PRICE_TP", "1")))  # 1=ON, 0=OFF
    hard_tp_use_mark: bool = bool(int(os.getenv("HARD_TP_USE_MARK", "0")))  # 1=mark, 0=last


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
    cfg.basis_stop_delta_bps = float(os.getenv("BASIS_STOP_DELTA_BPS", cfg.basis_stop_delta_bps))
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
        # Hỗ trợ cả camelCase và snake_case
        for name in ("fetchFundingHistory", "fetch_funding_history"):
            fn = getattr(ex, name, None)
            if callable(fn):
                try:
                    return fn(symbol, since_ms, None) or []
                except Exception:
                    pass
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
    # Hỗ trợ cả camelCase và snake_case tùy phiên bản ccxt
    for name in ("fetchFundingRate", "fetch_funding_rate"):
        fn = getattr(ex, name, None)
        if callable(fn):
            try:
                d = fn(symbol) or {}
                if isinstance(d, dict) and d:
                    return d
            except Exception:
                pass
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


class TradeTracker:
    """Lưu trữ thời gian mở vị thế cho các symbol."""
    def __init__(self):
        self.entry_times: Dict[str, float] = {}  # symbol -> timestamp
        self.entry_prices: Dict[str, Dict[str, float]] = {}  # symbol -> {ex_id -> price}
    
    def record_entry(self, symbol: str, binance_price: Optional[float] = None, bybit_price: Optional[float] = None):
        """Ghi lại thời gian và giá entry khi mở vị thế mới."""
        self.entry_times[symbol] = time.time()
        if symbol not in self.entry_prices:
            self.entry_prices[symbol] = {}
        if binance_price is not None:
            self.entry_prices[symbol]["binance"] = binance_price
        if bybit_price is not None:
            self.entry_prices[symbol]["bybit"] = bybit_price
    
    def get_hold_minutes(self, symbol: str) -> float:
        """Lấy số phút đã giữ vị thế."""
        entry_time = self.entry_times.get(symbol)
        if entry_time is None:
            return 0.0
        return (time.time() - entry_time) / 60.0
    
    def get_entry_prices(self, symbol: str) -> Dict[str, float]:
        """Lấy giá entry của symbol."""
        return self.entry_prices.get(symbol, {})

    def clear_symbol(self, symbol: str):
        """Xóa thông tin của một symbol."""
        if symbol in self.entry_times:
            del self.entry_times[symbol]
        if symbol in self.entry_prices:
            del self.entry_prices[symbol]

# Khởi tạo global tracker
TRADE_TRACKER = TradeTracker()

def calculate_pnl_pct(bin_entry: float, bin_current: float, byb_entry: float, byb_current: float, 
                     bin_side: str, leverage: int) -> float:
    """
    Tính % lãi/lỗ của cặp vị thế cross-exchange.
    
    Args:
        bin_entry: Giá entry trên Binance
        bin_current: Giá hiện tại trên Binance
        byb_entry: Giá entry trên Bybit
        byb_current: Giá hiện tại trên Bybit
        bin_side: Phía Binance ("long" hoặc "short")
        leverage: Đòn bẩy
        
    Returns:
        % lãi/lỗ (đã tính leverage)
    """
    if bin_side == "long":  # Binance long, Bybit short
        pnl_pct = ((bin_current / bin_entry - 1.0) - (byb_current / byb_entry - 1.0)) * 100 * leverage
    else:  # Binance short, Bybit long
        pnl_pct = ((1.0 - bin_current / bin_entry) - (1.0 - byb_current / byb_entry)) * 100 * leverage
    
    return pnl_pct

    

# Sửa hàm place_delta_neutral để ghi nhận thời gian entry
def place_delta_neutral(bin_short: bool, notional_usdt: float, symbol: str, binance, bybit, 
                        dry_run: bool, mode: str, cfg: Config, logger: Optional[TradeLogger] = None,
                        meta: Dict[str, Any] = None):
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
            # === TAKER: đặt 2 chân theo kiểu "atomic" với rollback ===
        amtA = _qty_from_notional(exA, symbol, notional_usdt, pA)
        amtB = _qty_from_notional(exB, symbol, notional_usdt, pB)

        if mode == "taker":
            # 1) Gửi chân A
            try:
                oA = _safe_create_order(exA, symbol, "market", sideA, amtA)
            except Exception as e:
                console.print(f"[red]Leg-A failed ({exA.id}): {e}[/red]")
                if logger:
                    logger.log({**log_base, "action": "legA_failed", "exA": exA.id, "exB": exB.id, "error": str(e)})
                return  # chưa mở chân nào, rút

            # 2) Gửi chân B; nếu fail → rollback A (reduceOnly)
            try:
                oB = _safe_create_order(exB, symbol, "market", sideB, amtB)
            except Exception as e:
                console.print(f"[red]Leg-B failed ({exB.id}), rolling back leg-A: {e}[/red]")
                if logger:
                    logger.log({**log_base, "action": "legB_failed_rollback", "orderIdA": oA.get("id", ""), "error": str(e)})
                # rollback: đóng ngay chân A (đối ứng, reduceOnly)
                rb_sideA = "sell" if sideA == "buy" else "buy"
                params = {"reduceOnly": True}
                if getattr(exA, "id", "") == "bybit":
                    params["timeInForce"] = "IOC"
                try:
                    _safe_create_order(exA, symbol, "market", rb_sideA, amtA, None, {"reduceOnly": True})
                except Exception as e2:
                    console.print(f"[red]Rollback failed on {exA.id}: {e2}[/red]")
                    if logger:
                        logger.log({**log_base, "action": "rollback_failed", "error": str(e2)})
                return  # kết thúc vì đã rollback xong

            # 3) Cả hai chân đã vào thành công
            if logger:
                logger.log({
                    **log_base, "action": "placed_taker",
                    "amountA": f"{amtA}", "amountB": f"{amtB}",
                    "orderIdA": oA.get("id",""), "orderIdB": oB.get("id","")
                })
            console.print("[green]Placed taker orders on both legs[/green]")
            try:
                STATE.entry_count += 1
                if TELE:
                    TELE.send(f"ENTER {symbol}: {exA.id} {sideA}/{exB.id} {sideB} notional≈{notional_usdt:.0f} (taker)")
            except Exception:
                pass
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
            try:
                STATE.entry_count += 1
                if TELE:
                    TELE.send(f"ENTER {symbol}: maker post-only {exA.id}/{exB.id} notional≈{notional_usdt:.0f}")
            except Exception:
                pass
            return

        # HYBRID wait and evaluate
        time.sleep(max(1, int(cfg.hybrid_wait_seconds)))
        rA = _order_filled_ratio(exA, oA)
        rB = _order_filled_ratio(exB, oB)
        logger.log({**log_base, "action": "hybrid_check", "fill_ratio_A": f"{rA:.4f}", "fill_ratio_B": f"{rB:.4f}"})
        if rA >= cfg.hybrid_min_fill_ratio and rB >= cfg.hybrid_min_fill_ratio:
            console.print("[green]Hybrid maker filled sufficiently. Keeping maker orders.[/green]")
            logger.log({**log_base, "action": "hybrid_keep_maker", "fill_ratio_A": f"{rA:.4f}", "fill_ratio_B": f"{rB:.4f}"})
            try:
                STATE.entry_count += 1
                if TELE:
                    TELE.send(f"ENTER {symbol}: maker-filled rA={rA:.2f}, rB={rB:.2f}")
            except Exception:
                pass
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
        try:
            STATE.entry_count += 1
            if TELE:
                TELE.send(f"ENTER {symbol}: hybrid complete rem≈{rem:.0f} USDT")
        except Exception:
            pass

    except Exception as e:
        console.print(f"[red]Order placement failed: {e}[/red]")
        logger.log({**log_base, "action": "error", "exA": exA.id, "exB": exB.id})
        try:
            if TELE:
                TELE.send(f"ORDER ERROR {symbol}: {e}")
            STATE.last_error = str(e)
        except Exception:
            pass

def close_delta_neutral(symbol: str, binance, bybit, dry_run: bool = False):
    _close_leg_reduce_only(binance, symbol)
    _close_leg_reduce_only(bybit,   symbol)
    console.print(f"[green]Closed delta-neutral positions for {symbol}[/green]")
    # Xóa khỏi tracker
    TRADE_TRACKER.clear_symbol(symbol)
    # notify + telemetry
    try:
        STATE.exit_count += 1
        if TELE:
            TELE.send(f"EXIT {symbol}: reduce-only both legs")
    except Exception:
        pass

def _close_leg_reduce_only(ex, symbol: str):
    if not has_creds(ex):
        return
    try:
        positions = ex.fetch_positions([symbol])
    except Exception:
        positions = []
    for p in positions:
        if p.get("symbol") != symbol:
            continue
        contracts = _to_float(p.get("contracts"))
        side = _position_side(p, contracts)
        params = {"reduceOnly": True}
        if side == "long" and contracts > 0:
            _safe_create_order(ex, symbol, "market", "sell", abs(contracts), None, params)
        elif side == "short" and contracts < 0:
            _safe_create_order(ex, symbol, "market", "buy",  abs(contracts), None, params)


# ===== Helpers for Telegram summaries and control =====
def _fmt_usdt(x: float) -> str:
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return str(x)

def _balances_summary(binance, bybit) -> str:
    try:
        b = binance.fetch_balance()
    except Exception:
        b = {}
    try:
        y = bybit.fetch_balance()
    except Exception:
        y = {}
    def pick(d):
        u = d.get("USDT") or {}
        return _to_float(u.get("total")), _to_float(u.get("free")), _to_float(u.get("used"))
    bt, bf, bu = pick(b); yt, yf, yu = pick(y)
    return (
        f"BINANCE — total:{_fmt_usdt(bt)} free:{_fmt_usdt(bf)} used:{_fmt_usdt(bu)}\n"
        f"BYBIT   — total:{_fmt_usdt(yt)} free:{_fmt_usdt(yf)} used:{_fmt_usdt(yu)}"
    )

def _positions_summary(ex) -> List[str]:
    out = []
    try:
        poss = ex.fetch_positions() or []
    except Exception:
        poss = []
    for p in poss:
        contracts = _to_float(p.get("contracts"))
        if contracts == 0:
            continue
        sym = p.get("symbol")
        side = _position_side(p, contracts)
        entry = _to_float(p.get("entryPrice"))
        mark  = _to_float(p.get("markPrice")) or _to_float(p.get("last"))
        notion = _to_float(p.get("notional"))
        out.append(f"{ex.id.upper()} {sym} {side} {contracts:g} @ {entry} mark {mark} notion≈{_fmt_usdt(notion)}")
    return out or [f"{ex.id.upper()} None"]

def _pnl_summary(ex) -> List[str]:
    out = []
    try:
        poss = ex.fetch_positions() or []
    except Exception:
        poss = []
    for p in poss:
        contracts = _to_float(p.get("contracts"))
        if contracts == 0:
            continue
        sym = p.get("symbol")
        entry = _to_float(p.get("entryPrice"))
        mark  = _to_float(p.get("markPrice")) or _to_float(p.get("last"))
        try:
            pnl = float(p.get("unrealizedPnl"))
        except Exception:
            pnl = (mark - entry) * contracts
        out.append(f"{ex.id.upper()} {sym} uPnL≈{_fmt_usdt(pnl)} (entry {entry}, mark {mark}, qty {contracts:g})")
    return out or [f"{ex.id.upper()} No open positions"]

def _status_text(cfg: Config, binance, bybit) -> str:
    open_syms = (_open_symbols(binance) | _open_symbols(bybit))
    dyn = compute_dynamic_notional_usdt(cfg, binance, bybit) if cfg.use_dynamic_notional else cfg.notional_usdt
    uptime_min = int((time.time() - STATE.started_ts)/60)
    lines = [
        f"Mode: {'MULTIPAIR' if cfg.multipair else 'SINGLE'} {cfg.execution_mode.upper()}, dry_run={cfg.dry_run}",
        f"Uptime: {uptime_min} phút",
        f"Open pairs: {len(open_syms)} | Entries={STATE.entry_count} Exits={STATE.exit_count} | Paused={STATE.paused}",
        f"Notional: cơ bản={cfg.notional_usdt} dyn≈{_fmt_usdt(dyn)} (lev {cfg.leverage}x, pct {cfg.notional_pct}, buf {int(cfg.reserve_buffer_pct*100)}%)",
        f"Scan: minVol={cfg.min_volume_usdt}, depthMult={cfg.min_depth_multiplier}, within={cfg.depth_within_bps} bps, maxH={cfg.max_hours_left}",
    ]
    lines.append(f"Edge: min={cfg.min_edge_bps:.1f} | TP={cfg.take_profit_edge_bps:.1f} | Basis max={cfg.max_basis_bps:.1f} | Basis stop Δ={cfg.basis_stop_delta_bps:.1f}")
    lines.append(f"Price TP: {cfg.price_arbitrage_tp_pct:.1f}% | SL: {cfg.price_arbitrage_sl_pct:.1f}% | Min hold: {cfg.price_arbitrage_min_hold_minutes} min | Funding countdown: {cfg.funding_countdown_minutes} min")
    return "\n".join(lines)

def _set_pause(flag: bool) -> str:
    STATE.paused = bool(flag)
    return f"Paused={STATE.paused}"

def _close_leg_reduce_only_single(ex, symbol: str):
    try:
        poss = ex.fetch_positions([symbol]) or []
    except Exception as e:
        return f"{ex.id} fetch_positions error: {e}"
    for p in poss:
        if p.get("symbol") != symbol:
            continue
        contracts = _to_float(p.get("contracts"))
        if contracts == 0:
            return f"{ex.id} {symbol}: no position"
        side = _position_side(p, contracts)
        amt = abs(contracts)
        try:
            if side == "long":
                ex.create_order(symbol, "market", "sell", amt, None, {"reduceOnly": True})
            else:
                ex.create_order(symbol, "market", "buy", amt, None, {"reduceOnly": True})
            return f"{ex.id} {symbol}: closed {side} {amt:g}"
        except Exception as e:
            return f"{ex.id} {symbol} close failed: {e}"
    return f"{ex.id} {symbol}: not found"

def _close_cmd(arg: str, binance, bybit) -> str:
    sym = (arg or "").strip()
    if not sym:
        return "Usage: /close SYMBOL (vd: BTC/USDT:USDT)"
    try:
        close_delta_neutral(sym, binance, bybit)
        return f"Closed both legs {sym}"
    except Exception as e:
        return f"Close failed: {e}"

def _closeleg_cmd(arg: str, binance, bybit) -> str:
    parts = (arg or "").split()
    if len(parts) < 2:
        return "Usage: /closeleg BINANCEUSDM|BYBIT SYMBOL"
    ex_name = parts[0].strip().lower()
    sym = " ".join(parts[1:]).strip()
    if ex_name.startswith("binance"):
        return _close_leg_reduce_only_single(binance, sym)
    if ex_name.startswith("bybit"):
        return _close_leg_reduce_only_single(bybit, sym)
    return "Exchange must be BINANCEUSDM or BYBIT"

def _set_cmd_multi(arg: str, cfg: Config, binance, bybit) -> str:
    """Hỗ trợ /set KEY VALUE hoặc /set KEY=VAL KEY2=VAL2 ..."""
    allowed = {
        "MIN_EDGE_BPS": ("min_edge_bps", float),
        "TAKE_PROFIT_EDGE_BPS": ("take_profit_edge_bps", float),
        "MAX_BASIS_BPS": ("max_basis_bps", float),
        "BASIS_STOP_DELTA_BPS": ("basis_stop_delta_bps", float),
        "MAX_HOURS_LEFT": ("max_hours_left", float),
        "SPIKE_MIN_EDGE_BPS": ("spike_min_edge_bps", float),
        "LEVERAGE": ("leverage", int),
        "USE_DYNAMIC_NOTIONAL": ("use_dynamic_notional", "bool"),
        "NOTIONAL_PCT": ("notional_pct", float),
        "RESERVE_BUFFER_PCT": ("reserve_buffer_pct", float),
        "MIN_NOTIONAL_USDT": ("min_notional_usdt", float),
        "MAX_NOTIONAL_USDT": ("max_notional_usdt", float),
    }
    if not arg:
        return "Usage: /set KEY VALUE | hoặc /set KEY=VAL KEY2=VAL2 ..."
    pairs = []
    tokens = arg.replace("\n", " ").split()
    if len(tokens) == 2 and "=" not in tokens[0] and "=" not in tokens[1]:
        pairs.append((tokens[0], tokens[1]))
    else:
        for tok in tokens:
            if "=" in tok:
                k, v = tok.split("=", 1)
                pairs.append((k, v))
    if not pairs:
        return "Usage: /set KEY VALUE | hoặc /set KEY=VAL KEY2=VAL2 ..."

    changes = []
    leverage_changed = None
    for k, v in pairs:
        K = k.strip().upper()
        if K not in allowed:
            changes.append(f"Skip {K}: not allowed")
            continue
        attr, typ = allowed[K]
        try:
            if     typ == int:   newv = int(float(v))
            elif   typ == float: newv = float(v)
            elif   typ == "bool": newv = _to_bool(v, False)
            else:                 newv = v
            setattr(cfg, attr, newv)
            changes.append(f"{K} -> {getattr(cfg, attr)}")
            if K == "LEVERAGE":
                leverage_changed = newv
        except Exception as e:
            changes.append(f"{K} failed: {e}")

    if leverage_changed is not None:
        for sym in (_open_symbols(binance) | _open_symbols(bybit)):
            try:
                ensure_leverage(binance, sym, leverage_changed, cfg.dry_run)
                ensure_leverage(bybit,   sym, leverage_changed, cfg.dry_run)
            except Exception:
                pass

    return "Updated:\n" + "\n".join(changes)

def _open_symbols(ex) -> Set[str]:
    """Danh sách symbol đang có vị thế (contracts != 0) trên 1 sàn."""
    try:
        poss = ex.fetch_positions() or []
    except Exception:
        poss = []
    out: Set[str] = set()
    for p in poss:
        try:
            contracts = _to_float(p.get("contracts"))
            sym = p.get("symbol")
            if sym and abs(contracts) > 0:
                out.add(sym)
        except Exception:
            continue
    return out

def _send_menu() -> str:
    # Menu đơn giản dạng text
    return (
        "Commands:\n"
        "/status | /balances (/balance) | /positions | /pnl (/profit)\n"
        "/openpairs | /pause | /resume | /shutdown | /count\n"
        "/set KEY=VAL ...\n"
        "/close SYMBOL | /closeleg BINANCEUSDM|BYBIT SYMBOL\n"
        "/help"
    )

def _wire_telegram(cfg: Config, binance, bybit):
    # ...existing code lấy token/chat id...
    global TELE
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id_env = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    allowed_ids: Set[int] = set()
    default_chat = None
    if chat_id_env:
        for part in chat_id_env.replace(";", ",").split(","):
            s = part.strip()
            if s.isdigit():
                n = int(s); allowed_ids.add(n)
                if default_chat is None: default_chat = n
    TELE = TelegramManager(token, allowed_ids, default_chat)

    # các lệnh có sẵn
    TELE.on("/help",      lambda a, c: _send_menu())
    TELE.on("/menu",      lambda a, c: _send_menu())
    TELE.on("/start",     lambda a, c: (_set_pause(False) or _send_menu()))
    TELE.on("/stop",      lambda a, c: _set_pause(True))
    TELE.on("/count",     lambda a, c: f"Entries={STATE.entry_count} Exits={STATE.exit_count} Errors={STATE.last_error or 'None'}")
    TELE.on("/balance",   lambda a, c: _balances_summary(binance, bybit))
    TELE.on("/profit",    lambda a, c: "\n".join(_pnl_summary(binance) + _pnl_summary(bybit)))
    TELE.on("/status",    lambda a, c: _status_text(cfg, binance, bybit))
    TELE.on("/balances",  lambda a, c: _balances_summary(binance, bybit))
    TELE.on("/positions", lambda a, c: "\n".join(_positions_summary(binance) + _positions_summary(bybit)))
    TELE.on("/pnl",       lambda a, c: "\n".join(_pnl_summary(binance) + _pnl_summary(bybit)))
    TELE.on("/openpairs", lambda a, c: ", ".join(sorted(_open_symbols(binance) | _open_symbols(bybit))) or "None")
    TELE.on("/pause",     lambda a, c: _set_pause(True))
    TELE.on("/resume",    lambda a, c: _set_pause(False))

    # FIX: wire /set và alias
    TELE.on("/set",       lambda a, c: _set_cmd_multi(a, cfg, binance, bybit))
    TELE.on("set",        lambda a, c: _set_cmd_multi(a, cfg, binance, bybit))   # alias không có '/'

    TELE.on("/close",     lambda a, c: _close_cmd(a, binance, bybit))
    TELE.on("/closeleg",  lambda a, c: _closeleg_cmd(a, binance, bybit))
    TELE.on("/shutdown",  lambda a, c: _shutdown_cmd())
    TELE.on("default",    lambda a, c: "Unknown. /help")

    TELE.start()

    # log danh sách lệnh để kiểm tra nhanh
    print("Telegram commands:", ", ".join(sorted(TELE.handlers.keys())))
    if TELE.default_chat_id:
        TELE.send("Bot online.\n" + _status_text(cfg, binance, bybit))

# --- Shutdown command ---
def _shutdown_cmd() -> str:
    """Thoát tiến trình sau khi gửi phản hồi Telegram."""
    global RUNNING
    RUNNING = False
    def _quit():
        try:
            if TELE:
                TELE.stop()
        except Exception:
            pass
        os._exit(0)  # thoát ngay, mã 0
    threading.Timer(1.0, _quit).start()  # đợi 1s cho Telegram gửi xong
    return "Shutting down..."

# === main() ===
def main():
    cfg = load_config()
    print_env_status()
    binance, bybit = init_exchanges()
    logger = TradeLogger(cfg)
    funder = FundingTracker(cfg) if cfg.funding_collect else None
    _wire_telegram(cfg, binance, bybit)
    if cfg.multipair:
        multipair_scan_and_trade(binance, bybit, cfg, logger, funder)
    else:
        single_pair_loop(binance, bybit, cfg, logger, funder)  # Sửa dấu ngoặc

def single_pair_loop(binance, bybit, cfg: Config, logger: TradeLogger, funder: Optional[FundingTracker]):
    global RUNNING
    symbol = cfg.symbol
    last_h: Optional[float] = None
    weak_hits = 0
    while RUNNING:
        STATE.last_loop_ts = time.time()
        if STATE.paused:
            time.sleep(3)
            continue
        try:
            # Prices + basis
            t_bin = binance.fetch_ticker(symbol); t_byb = bybit.fetch_ticker(symbol)
            p_bin = _to_float(t_bin.get("last")) or _to_float(t_bin.get("mark")) or _to_float(t_bin.get("close"))
            p_byb = _to_float(t_byb.get("last")) or _to_float(t_byb.get("mark")) or _to_float(t_byb.get("close"))
            basis_bps = get_basis_bps(p_bin, p_byb)

            # Funding + edge
            fr_b, h_b = read_next_funding(binance, symbol)
            fr_y, h_y = read_next_funding(bybit,   symbol)
            h_left = min(h_b, h_y) if (h_b > 0 and h_y > 0) else (max(h_b, h_y) if (h_b > 0 or h_y > 0) else _hours_to_next_8h_utc(binance.milliseconds()))
            eA = funding_edge_now_bps(fr_b, fr_y, h_left); eB = funding_edge_now_bps(fr_y, fr_b, h_left)
            bin_short = eA >= eB
            best_edge = eA if bin_short else eB

            entry_fee_bps = 0.0 if cfg.execution_mode in ("maker","hybrid") else (taker_bps(binance, symbol) + taker_bps(bybit, symbol))
            need = max(cfg.min_edge_bps, entry_fee_bps)
            score = best_edge - need

            header = f"Cross-Exchange Funding Bot | lev={cfg.leverage}x | notional={cfg.notional_usdt:.1f} USDT | dry_run={cfg.dry_run} | mode={cfg.execution_mode} | maxH={cfg.max_hours_left}h spike>={cfg.spike_min_edge_bps}bps"
            console.print(Panel.fit(header, style="bold white on blue"))

            # Entry
            if (score >= 0) and (h_left <= cfg.max_hours_left) and (abs(basis_bps) <= cfg.max_basis_bps):
                ensure_leverage(binance, symbol, cfg.leverage, cfg.dry_run)
                ensure_leverage(bybit,   symbol, cfg.leverage, cfg.dry_run)
                dyn_notional = compute_dynamic_notional_usdt(cfg, binance, bybit)
                meta = {"edge_best": best_edge, "need": need, "hours_left": h_left, "basis_bps": basis_bps}
                place_delta_neutral(bin_short, dyn_notional, symbol, binance, bybit, cfg.dry_run, cfg.execution_mode, cfg, logger, meta)
            else:
                console.print(f"[yellow]Hold: score={score:.3f}, basis={basis_bps:.2f} bps, h_left={h_left:.2f}[/yellow]")

            # Auto-exit cho symbol này nếu đang mở
            open_syms = (_open_symbols(binance) | _open_symbols(bybit))
            if symbol in open_syms:
                if cfg.hard_price_tp:
                    try:
                        # Lấy vị thế hiện tại ở 2 sàn
                        positions_bin = _get_positions_by_symbol(binance, symbol)
                        positions_byb = _get_positions_by_symbol(bybit, symbol)

                        if positions_bin and positions_byb:
                            pos_bin = positions_bin[0]
                            pos_byb = positions_byb[0]

                            # Xác định bên long/short theo contracts ở Binance
                            bin_contracts = _to_float(pos_bin.get("contracts", 0))
                            bin_side = "long" if bin_contracts > 0 else "short"

                            # Lấy entry (ưu tiên từ vị thế; fallback về giá lúc vào do tracker lưu)
                            entry_prices = TRADE_TRACKER.get_entry_prices(symbol)
                            bin_entry = _to_float(pos_bin.get("entryPrice")) or entry_prices.get("binance", p_bin)
                            byb_entry = _to_float(pos_byb.get("entryPrice")) or entry_prices.get("bybit",  p_byb)

                            # Giá hiện tại đã có: p_bin, p_byb (last/mark/close fallback sẵn)
                            pnl_pct = calculate_pnl_pct(
                                bin_entry=bin_entry, bin_current=p_bin,
                                byb_entry=byb_entry, byb_current=p_byb,
                                bin_side=bin_side, leverage=cfg.leverage
                            )

                            if pnl_pct >= cfg.price_arbitrage_tp_pct:
                                console.print(f"[green][HARD-TP] {symbol}: {pnl_pct:.2f}% ≥ {cfg.price_arbitrage_tp_pct}% → closing now[/green]")
                                close_delta_neutral(symbol, binance, bybit)
                                continue  # sang symbol kế tiếp sau khi đã đóng
                    except Exception as e:
                        console.print(f"[red][HARD-TP] error for {symbol}: {e}[/red]")

                # Kiểm tra funding sắp diễn ra không
                funding_soon = h_left <= (cfg.funding_countdown_minutes / 60.0)
                
                # 0) Nếu funding sắp diễn ra (< 30 phút), ưu tiên đợi để thu funding
                if funding_soon:
                    console.print(f"[cyan]Funding soon for {symbol}: {h_left:.2f}h left, holding position[/cyan]")
                else:
                    # Kiểm tra price arbitrage TP/SL nếu đã giữ đủ lâu
                    hold_minutes = TRADE_TRACKER.get_hold_minutes(symbol)
                    if hold_minutes >= cfg.price_arbitrage_min_hold_minutes:
                        try:
                            # Lấy thông tin vị thế để xác định bên nào long/short
                            positions_bin = _get_positions_by_symbol(binance, symbol)
                            positions_byb = _get_positions_by_symbol(bybit, symbol)
                            
                            if positions_bin and positions_byb:
                                pos_bin = positions_bin[0]
                                pos_byb = positions_byb[0]
                                
                                bin_contracts = _to_float(pos_bin.get("contracts", 0))
                                bin_side = "long" if bin_contracts > 0 else "short"
                                
                                # Lấy giá entry từ vị thế hoặc từ tracker
                                entry_prices = TRADE_TRACKER.get_entry_prices(symbol)
                                bin_entry = _to_float(pos_bin.get("entryPrice")) or entry_prices.get("binance", p_bin)
                                byb_entry = _to_float(pos_byb.get("entryPrice")) or entry_prices.get("bybit", p_byb)
                                
                                # Tính P&L %
                                pnl_pct = calculate_pnl_pct(bin_entry, p_bin, byb_entry, p_byb, bin_side, cfg.leverage)
                                
                                # StopLoss
                                if pnl_pct <= cfg.price_arbitrage_sl_pct:
                                    console.print(f"[red]StopLoss triggered for {symbol}: {pnl_pct:.2f}% ≤ {cfg.price_arbitrage_sl_pct}% → closing[/red]")
                                    close_delta_neutral(symbol, binance, bybit)
                                    continue
                                
                                # TakeProfit
                                if pnl_pct >= cfg.price_arbitrage_tp_pct:
                                    console.print(f"[green]Price arbitrage TP for {symbol}: {pnl_pct:.2f}% ≥ {cfg.price_arbitrage_tp_pct}% → closing[/green]")
                                    close_delta_neutral(symbol, binance, bybit)
                                    continue
                                else:
                                    console.print(f"[blue]Current P&L for {symbol}: {pnl_pct:.2f}%, held for {hold_minutes:.1f} min[/blue]")
                        except Exception as e:
                            console.print(f"[yellow]Price arbitrage check error on {symbol}: {e}[/yellow]")

            # Auto-exit cho symbol này nếu đang mở
            open_syms = (_open_symbols(binance) | _open_symbols(bybit))
            if symbol in open_syms:
                # 1) Sau mốc funding
                if last_h is not None and last_h < 0.10 and h_left > 7.5:
                    close_delta_neutral(symbol, binance, bybit)
                last_h = h_left
                # 2) TP khi edge yếu/score âm (debounce)
                weak_now = (best_edge <= cfg.take_profit_edge_bps) or (score < 0)
                weak_hits = weak_hits + 1 if weak_now else 0
                if weak_hits >= 2:
                    close_delta_neutral(symbol, binance, bybit)
                    weak_hits = 0
                # 3) Basis stop
                if abs(basis_bps) > (cfg.max_basis_bps + cfg.basis_stop_delta_bps):
                    console.print(f"[red]Basis stop: |{basis_bps:.2f}| > {cfg.max_basis_bps}+{cfg.basis_stop_delta_bps} → closing {symbol}[/red]")
                    close_delta_neutral(symbol, binance, bybit)

            if funder:
                funder.maybe_snapshot(cfg, binance, bybit, symbol)

        except KeyboardInterrupt:
            console.print("[red]Ctrl+C detected. Exiting loop.[/red]")
            break
        except Exception as e:
            console.print(f"[red]Loop error: {e}[/red]")
            STATE.last_error = str(e)
        time.sleep(max(1, int(cfg.poll_seconds)))

def multipair_scan_and_trade(binance, bybit, cfg: Config, logger: TradeLogger, funder: Optional[FundingTracker]):
    global RUNNING
    last_hours: Dict[str, float] = {}
    decay_hits: Dict[str, int] = {}
    decay_needed = 2
    while RUNNING:
        STATE.last_loop_ts = time.time()
        if STATE.paused:
            time.sleep(3)
            continue
        try:
            header = f"Cross-Exchange Funding Bot | lev={cfg.leverage}x | notional={cfg.notional_usdt:.1f} USDT | dry_run={cfg.dry_run} | mode={cfg.execution_mode} | multipair=ON | maxH={cfg.max_hours_left}h spike>={cfg.spike_min_edge_bps}bps"
            console.print(Panel.fit(header, style="bold white on blue"))

            syms = common_linear_usdt_symbols(binance, bybit)
            if cfg.allowlist:
                syms = [s for s in syms if s in set(cfg.allowlist)]
            if cfg.denylist:
                deny = set(cfg.denylist)
                syms = [s for s in syms if s not in deny]

            dyn_notional = compute_dynamic_notional_usdt(cfg, binance, bybit)
            dyn_min_vol = max(cfg.min_volume_usdt, cfg.volume_x_notional * dyn_notional)
            min_depth_usdt = cfg.min_depth_multiplier * dyn_notional

            rows: List[Dict[str, Any]] = []
            for s in syms:
                try:
                    vol_bin = get_quote_volume_usdt(binance, s)
                    vol_byb = get_quote_volume_usdt(bybit,   s)
                    if vol_bin < dyn_min_vol or vol_byb < dyn_min_vol:
                        continue
                    d_bid_bin, d_ask_bin = orderbook_depth_usdt(binance, s, cfg.depth_within_bps)
                    d_bid_byb, d_ask_byb = orderbook_depth_usdt(bybit,   s, cfg.depth_within_bps)
                    if min(d_bid_bin, d_ask_bin, d_bid_byb, d_ask_byb) < min_depth_usdt:
                        continue

                    fr_b, h_b = read_next_funding(binance, s)
                    fr_y, h_y = read_next_funding(bybit,   s)
                    h_left = min(h_b, h_y) if (h_b > 0 and h_y > 0) else (max(h_b, h_y) if (h_b > 0 or h_y > 0) else _hours_to_next_8h_utc(binance.milliseconds()))
                    if h_left > cfg.max_hours_left:
                        continue

                    t_bin = binance.fetch_ticker(s); t_byb = bybit.fetch_ticker(s)
                    p_bin = _to_float(t_bin.get("last")) or _to_float(t_bin.get("mark")) or _to_float(t_bin.get("close"))
                    p_byb = _to_float(t_byb.get("last")) or _to_float(t_byb.get("mark")) or _to_float(t_byb.get("close"))
                    basis = get_basis_bps(p_bin, p_byb)
                    if abs(basis) > cfg.max_basis_bps:
                        continue

                    eA = funding_edge_now_bps(fr_b, fr_y, h_left); eB = funding_edge_now_bps(fr_y, fr_b, h_left)
                    edge = max(eA, eB)
                    dir_str = "BIN short" if eA >= eB else "BYB short"
                    entry_fee_bps = 0.0 if cfg.execution_mode in ("maker","hybrid") else (taker_bps(binance, s) + taker_bps(bybit, s))
                    need = max(cfg.min_edge_bps, entry_fee_bps)
                    score = edge - need
                    if edge < cfg.spike_min_edge_bps:
                        continue
                    rows.append({"symbol": s, "basis": basis, "edge": edge, "need": need, "score": score, "hours": h_left, "dir": dir_str, "vol_bin": vol_bin, "vol_byb": vol_byb})
                except Exception:
                    continue

            rows.sort(key=lambda r: r["score"], reverse=True)
            top = rows[:cfg.max_pairs_to_show]

            table = Table(title=f"Multipair scan (minVol≈{dyn_min_vol:,.0f} USDT, depth≥{min_depth_usdt:,.0f} within ±{cfg.depth_within_bps} bps)", box=box.SIMPLE_HEAVY)
            table.add_column("#", justify="right"); table.add_column("Symbol"); table.add_column("Basis bps", justify="right"); table.add_column("Edge bps", justify="right"); table.add_column("Need", justify="right"); table.add_column("Score", justify="right"); table.add_column("H left", justify="right"); table.add_column("Dir"); table.add_column("Vol BIN", justify="right"); table.add_column("Vol BYB", justify="right")
            for i, r in enumerate(top, 1):
                table.add_row(str(i), r["symbol"], f"{r['basis']:.2f}", f"{r['edge']:.3f}", f"{r['need']:.3f}", f"{r['score']:.3f}", f"{r['hours']:.2f}", r["dir"], f"{r['vol_bin']:.1f}", f"{r['vol_byb']:.1f}")
            console.print(table)

            if top and top[0]["score"] >= 0:
                best = top[0]
                bin_short = best["dir"].startswith("BIN")
                ensure_leverage(binance, best["symbol"], cfg.leverage, cfg.dry_run)
                ensure_leverage(bybit,   best["symbol"], cfg.leverage, cfg.dry_run)
                dyn_notional = compute_dynamic_notional_usdt(cfg, binance, bybit)
                meta = {"edge_best": best["edge"], "need": best["need"], "hours_left": best["hours"], "basis_bps": best["basis"]}
                place_delta_neutral(bin_short, dyn_notional, best["symbol"], binance, bybit, cfg.dry_run, cfg.execution_mode, cfg, logger, meta)
            else:
                console.print("[yellow]No candidate meets edge threshold[/yellow]")

            # Auto-exit cho TẤT CẢ các symbol đang mở
            open_syms = (_open_symbols(binance) | _open_symbols(bybit))
            for sym in list(open_syms):
                try:
                    t_bin = binance.fetch_ticker(sym); t_byb = bybit.fetch_ticker(sym)
                    p_bin = _to_float(t_bin.get("last")) or _to_float(t_bin.get("mark"))
                    p_byb = _to_float(t_byb.get("last")) or _to_float(t_byb.get("mark"))
                    basis = get_basis_bps(p_bin, p_byb)
                    fr_b, h_b = read_next_funding(binance, sym)
                    fr_y, h_y = read_next_funding(bybit,   sym)
                    h_left = min(h_b, h_y) if (h_b > 0 and h_y > 0) else (max(h_b, h_y) if (h_b > 0 or h_y > 0) else _hours_to_next_8h_utc(binance.milliseconds()))
                    eA = funding_edge_now_bps(fr_b, fr_y, h_left); eB = funding_edge_now_bps(fr_y, fr_b, h_left)
                    edge = max(eA, eB)
                    entry_fee_bps = 0.0 if cfg.execution_mode in ("maker","hybrid") else (taker_bps(binance, sym) + taker_bps(bybit, sym))
                    need = max(cfg.min_edge_bps, entry_fee_bps)
                    score = edge - need
                except Exception as e:
                    console.print(f"[red]Auto-exit calc failed on {sym}: {e}[/red]")
                    continue

                # Kiểm tra thời gian cần đợi funding
                funding_soon = h_left <= (cfg.funding_countdown_minutes / 60.0)
                
                # 1. Nếu funding sắp diễn ra (<30 phút), ưu tiên đợi để thu funding
                if funding_soon:
                    console.print(f"[cyan]Funding soon for {sym}: {h_left:.2f}h left, holding position[/cyan]")
                    continue
                    
                # 2. Kiểm tra TP theo giá (price arbitrage) nếu đã giữ đủ lâu
                hold_minutes = TRADE_TRACKER.get_hold_minutes(sym)
                if hold_minutes >= cfg.price_arbitrage_min_hold_minutes:
                    try:
                        # Lấy thông tin vị thế để xác định bên nào long/short
                        positions_bin = _get_positions_by_symbol(binance, sym)
                        positions_byb = _get_positions_by_symbol(bybit, sym)
                        
                        if positions_bin and positions_byb:
                            pos_bin = positions_bin[0]
                            pos_byb = positions_byb[0]
                            
                            bin_contracts = _to_float(pos_bin.get("contracts", 0))
                            bin_side = "long" if bin_contracts > 0 else "short"
                            
                            # Lấy giá entry từ vị thế hoặc từ tracker
                            entry_prices = TRADE_TRACKER.get_entry_prices(sym)
                            bin_entry = _to_float(pos_bin.get("entryPrice")) or entry_prices.get("binance", p_bin)
                            byb_entry = _to_float(pos_byb.get("entryPrice")) or entry_prices.get("bybit", p_byb)
                            
                            # Tính P&L %
                            pnl_pct = calculate_pnl_pct(bin_entry, p_bin, byb_entry, p_byb, bin_side, cfg.leverage)
                            
                            # THÊM: StopLoss
                            if pnl_pct <= cfg.price_arbitrage_sl_pct:
                                console.print(f"[red]StopLoss triggered for {sym}: {pnl_pct:.2f}% ≤ {cfg.price_arbitrage_sl_pct}% → closing[/red]")
                                close_delta_neutral(sym, binance, bybit)
                                continue
                            
                            # TakeProfit (đã có)
                            if pnl_pct >= cfg.price_arbitrage_tp_pct:
                                console.print(f"[green]Price arbitrage TP for {sym}: {pnl_pct:.2f}% ≥ {cfg.price_arbitrage_tp_pct}% → closing[/green]")
                                close_delta_neutral(sym, binance, bybit)
                                continue

                            else:
                                console.print(f"[blue]Current P&L for {sym}: {pnl_pct:.2f}%, held for {hold_minutes:.1f} min[/blue]")
                    except Exception as e:
                        console.print(f"[yellow]Price arbitrage check error on {sym}: {e}[/yellow]")
                        
                # 3. Các điều kiện thoát khác vẫn giữ nguyên
                prev_h = last_hours.get(sym); last_hours[sym] = h_left
                if prev_h is not None and prev_h < 0.10 and h_left > 7.5:
                    close_delta_neutral(sym, binance, bybit)
                    continue
                    
                weak_now = (edge <= cfg.take_profit_edge_bps) or (score < 0)
                decay_hits[sym] = decay_hits.get(sym, 0) + 1 if weak_now else 0
                if decay_hits.get(sym, 0) >= decay_needed:
                    close_delta_neutral(sym, binance, bybit)
                    decay_hits[sym] = 0
                    continue
                    
                if abs(basis) > (cfg.max_basis_bps + cfg.basis_stop_delta_bps):
                    console.print(f"[red]Basis stop: |{basis:.2f}| > {cfg.max_basis_bps}+{cfg.basis_stop_delta_bps} → closing {sym}[/red]")
                    close_delta_neutral(sym, binance, bybit)
                    continue

        except KeyboardInterrupt:
            console.print("[red]Ctrl+C detected. Exiting loop.[/red]")
            break
        except Exception as e:
            console.print(f"[red]Multipair loop error: {e}[/red]")
            STATE.last_error = str(e)
        time.sleep(max(1, int(cfg.poll_seconds)))

def _get_positions_by_symbol(ex, symbol: str) -> List[Dict[str, Any]]:
    """Lấy thông tin vị thế cho một symbol cụ thể."""
    try:
        all_positions = ex.fetch_positions([symbol]) or []
        return [p for p in all_positions if p.get("symbol") == symbol and abs(_to_float(p.get("contracts", 0))) > 0]
    except Exception:
        return []

# Gọi main khi chạy trực tiếp
if __name__ == "__main__":
    main()
