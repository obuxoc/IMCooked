"""Algothon 2026 — Live Dashboard (Full-Featured).

Lightweight HTTP dashboard with:
  - PnL chart (live from API)
  - Price charts for all markets over time
  - Order book visualization with depth levels
  - Positions and exposure
  - ETF arb gap tracking
  - Volatility metrics per product
  - Signal approval/rejection stats
  - Risk state

Usage:
    from dashboard import Dashboard
    dash = Dashboard(bot, risk, executor, persist, port=8080)
    dash.start()  # opens http://localhost:8080
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algothon_bot import AlgothonBot
    from signals import RiskManager, Executor
    from data_cache import DataPersistence


# ── GLOBAL STATE ─────────────────────────────────────────────────────────
_bot: AlgothonBot | None = None
_risk: RiskManager | None = None
_executor: Executor | None = None
_persist: DataPersistence | None = None
_simulator = None  # DryRunSimulator when in dry-run mode
_alpha = None      # AlphaEngine for fair value display
# Rolling history for charts (stored server-side, sent to frontend)
_price_history: dict[str, deque] = {}       # product -> deque of {t, mid, bid, ask}
_arb_history: deque = deque(maxlen=500)      # deque of {t, gap, fair, etf_mid}
_pnl_from_api: deque = deque(maxlen=500)     # deque of {t, pnl}
_MAX_CHART_POINTS = 300

# Trade volume tracking — cumulative volume per product from trade stream
_trade_volume: dict[str, int] = {}           # product -> cumulative traded volume
_trade_count: dict[str, int] = {}            # product -> cumulative trade count
_last_trade_idx: int = 0                     # how many trades we've already counted


def _record_tick():
    """Called periodically to snapshot prices & PnL into chart histories."""
    if _bot is None:
        return

    now = time.time()

    # Price history per product
    for sym, snap in _bot.latest.items():
        if sym not in _price_history:
            _price_history[sym] = deque(maxlen=_MAX_CHART_POINTS)
        if not math.isnan(snap.mid):
            _price_history[sym].append({
                "t": now,
                "mid": round(snap.mid, 1),
                "bid": round(snap.best_bid, 1) if not math.isnan(snap.best_bid) else None,
                "ask": round(snap.best_ask, 1) if not math.isnan(snap.best_ask) else None,
            })

    # Arb gap history
    try:
        arb = _bot.cache.arb_snapshot()
        gap = arb["gap"]
        if not math.isnan(gap):
            _arb_history.append({"t": now, "gap": round(gap, 1),
                                  "fair": round(arb["fair_etf"], 1),
                                  "etf_mid": round(arb["etf_mid"], 1)})
    except Exception:
        pass

    # PnL from API
    if _bot:
        try:
            pnl_data = _bot.get_pnl()
            total = pnl_data.get("totalPnL", pnl_data.get("total", None))
            if total is not None:
                _pnl_from_api.append({"t": now, "pnl": round(float(total), 2)})
        except Exception:
            pass

    # Mark-to-market dry-run simulator
    if _simulator and _bot:
        try:
            _simulator.mark_to_market(_bot.cache.mids)
        except Exception:
            pass

    # Trade volume tracking — count new trades from cache
    global _last_trade_idx
    if _bot:
        try:
            trades = list(_bot.cache._trades)
            for t in trades[_last_trade_idx:]:
                prod = t.get("product", "")
                vol = t.get("volume", 0)
                _trade_volume[prod] = _trade_volume.get(prod, 0) + vol
                _trade_count[prod] = _trade_count.get(prod, 0) + 1
            _last_trade_idx = len(trades)
        except Exception:
            pass


class _Recorder(threading.Thread):
    """Background thread that snapshots data for charts every N seconds."""
    def __init__(self, interval: float = 3.0):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                _record_tick()
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


class _DashHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/api/state":
            self._serve_json(self._collect_state())
        elif self.path == "/api/history":
            self._serve_json(self._collect_full_history())
        else:
            self._serve_html()

    def _serve_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(_HTML_PAGE.encode())

    def _collect_full_history(self) -> dict:
        """Load ALL historical data from CSV files on disk."""
        import pandas as pd
        result: dict = {"price_history": {}, "trade_history": []}
        if not _persist:
            return result
        try:
            for product in _persist.get_all_products_on_disk():
                df = _persist.load_existing(product)
                if df.empty or "timestamp" not in df.columns:
                    continue
                df = df.sort_values("timestamp")
                points = []
                for _, row in df.iterrows():
                    pt = {"t": float(row["timestamp"]),
                          "mid": round(float(row["mid"]), 1) if "mid" in row and not (isinstance(row["mid"], float) and math.isnan(row["mid"])) else None}
                    if "best_bid" in row:
                        v = row["best_bid"]
                        pt["bid"] = round(float(v), 1) if not (isinstance(v, float) and math.isnan(v)) else None
                    if "best_ask" in row:
                        v = row["best_ask"]
                        pt["ask"] = round(float(v), 1) if not (isinstance(v, float) and math.isnan(v)) else None
                    points.append(pt)
                result["price_history"][product] = points
            # Trade history
            tdf = _persist.load_existing_trades()
            if not tdf.empty:
                tdf = tdf.sort_values("timestamp")
                trades = []
                for _, row in tdf.iterrows():
                    trades.append({
                        "t": float(row["timestamp"]),
                        "product": str(row.get("product", "")),
                        "price": float(row.get("price", 0)),
                        "volume": float(row.get("volume", 0)),
                        "buyer": str(row.get("buyer", "")) if not (isinstance(row.get("buyer"), float) and math.isnan(row.get("buyer"))) else "",
                        "seller": str(row.get("seller", "")) if not (isinstance(row.get("seller"), float) and math.isnan(row.get("seller"))) else "",
                    })
                result["trade_history"] = trades
        except Exception as e:
            result["error"] = str(e)
        return result

    def _collect_state(self) -> dict:
        state: dict = {"time": time.time(), "products": [], "orderbooks": {}}

        if _bot is None:
            return state

        # --- Per-product live data ---
        for sym, snap in _bot.latest.items():
            vol_val = 0.0
            try:
                from strategies import get_tracker
                vol_val = get_tracker().volatility(sym)
                ema_fast = get_tracker().ema_fast(sym)
                ema_slow = get_tracker().ema_slow(sym)
                trend = get_tracker().trend(sym)
            except Exception:
                ema_fast = ema_slow = trend = 0.0

            entry = {
                "product": sym,
                "mid": _safe(snap.mid),
                "best_bid": _safe(snap.best_bid),
                "best_ask": _safe(snap.best_ask),
                "spread": _safe(snap.spread),
                "imbalance": round(snap.imbalance, 3),
                "top_imbalance": round(snap.top_imbalance, 3),
                "micro_price": _safe(snap.micro_price),
                "weighted_mid": _safe(snap.weighted_mid),
                "bid_vol": snap.total_bid_vol,
                "ask_vol": snap.total_ask_vol,
                "bid_levels": snap.bid_levels,
                "ask_levels": snap.ask_levels,
                "position": 0,
                "volatility": round(vol_val * 10000, 2),  # in bps
                "ema_fast": _safe(ema_fast),
                "ema_slow": _safe(ema_slow),
                "trend": round(trend * 10000, 1) if isinstance(trend, float) else 0,
            }
            # Alpha fair values
            if _alpha:
                afv, acf = _alpha.get(sym)
                entry["alpha_fair"] = _safe(afv)
                entry["alpha_conf"] = round(acf * 100, 1) if acf else 0
            else:
                entry["alpha_fair"] = None
                entry["alpha_conf"] = 0
            state["products"].append(entry)

            # Order book depth (top 5 levels each side)
            ob_data = {"bids": [], "asks": []}
            for i, (p, v) in enumerate(zip(snap.bid_prices_3, snap.bid_vols_3)):
                if not math.isnan(p):
                    ob_data["bids"].append({"price": round(p, 1), "vol": v})
            for i, (p, v) in enumerate(zip(snap.ask_prices_3, snap.ask_vols_3)):
                if not math.isnan(p):
                    ob_data["asks"].append({"price": round(p, 1), "vol": v})
            state["orderbooks"][sym] = ob_data

        # Positions from risk manager
        if _risk:
            positions = _risk.positions
            for p in state["products"]:
                p["position"] = positions.get(p["product"], 0)

        # --- Risk stats ---
        if _risk:
            rs = _risk.get_stats()
            state["risk"] = {
                "signals_received": rs["signals_received"],
                "signals_approved": rs["signals_approved"],
                "signals_rejected": rs["signals_rejected"],
                "approval_rate": round(rs["approval_rate"] * 100, 1),
                "pnl": round(rs["current_pnl"], 2),
                "peak_pnl": round(rs["peak_pnl"], 2),
                "drawdown": round(rs["drawdown"], 2),
                "halted": rs["halted"],
                "halt_reason": rs["halt_reason"],
                "strategies_halted": rs["strategies_halted"],
                "rejection_breakdown": rs["rejection_breakdown"],
            }
        else:
            state["risk"] = {}

        # --- PnL from API (direct) ---
        state["pnl_history"] = list(_pnl_from_api)

        # --- Price history for charts ---
        state["price_history"] = {}
        for sym, hist in _price_history.items():
            state["price_history"][sym] = list(hist)[-_MAX_CHART_POINTS:]

        # --- Arb history ---
        state["arb_history"] = list(_arb_history)

        # --- Arb current ---
        try:
            arb = _bot.cache.arb_snapshot()
            from strategies import compute_fly_fair
            etf_ema = 0
            try:
                from strategies import get_tracker
                etf_ema = get_tracker().ema_slow("LON_ETF")
            except Exception:
                pass
            fly_fair = compute_fly_fair(etf_ema) if etf_ema and not math.isnan(etf_ema) else None
            state["arb"] = {
                "etf_mid": _safe(arb["etf_mid"]),
                "fair_etf": _safe(arb["fair_etf"]),
                "gap": _safe(arb["gap"]),
                "tide_mid": _safe(arb["tide_mid"]),
                "wx_mid": _safe(arb["wx_mid"]),
                "lhr_mid": _safe(arb["lhr_mid"]),
                "fly_mid": _safe(arb["fly_mid"]),
                "fly_fair": _safe(fly_fair),
            }
        except Exception:
            state["arb"] = {}

        # Active orders
        if _executor:
            state["active_orders"] = len(_executor._active_orders)
        else:
            state["active_orders"] = 0

        # Data files
        if _persist:
            state["data_files"] = _persist.get_all_products_on_disk()
        else:
            state["data_files"] = []

        # Dry-run simulator stats
        if _simulator:
            state["simulator"] = _simulator.get_stats()
        else:
            state["simulator"] = None

        # Trade volume per product
        state["trade_volume"] = dict(_trade_volume)
        state["trade_count"] = dict(_trade_count)

        # Alpha fair values summary
        if _alpha and _alpha.fair:
            state["alpha"] = {
                "fair": {k: _safe(v) for k, v in _alpha.fair.items()},
                "confidence": {k: round(v * 100, 1) for k, v in _alpha.confidence.items()},
                "hours_left": round(_alpha._hours_left(), 2),
                "last_update": round(_alpha.last_update, 1) if _alpha.last_update else None,
            }
        else:
            state["alpha"] = None

        return state


def _safe(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, float):
        return round(v, 2)
    return v


class Dashboard:
    def __init__(self, bot, risk, executor, persist=None, port: int = 8080,
                 simulator=None, alpha=None):
        global _bot, _risk, _executor, _persist, _simulator, _alpha
        _bot = bot
        _risk = risk
        _executor = executor
        _persist = persist
        _simulator = simulator
        _alpha = alpha
        self.port = port
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._recorder: _Recorder | None = None

    def start(self):
        self._server = HTTPServer(("0.0.0.0", self.port), _DashHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._recorder = _Recorder(interval=3.0)
        self._recorder.start()
        print(f"[DASHBOARD] Live at http://localhost:{self.port}")

    def stop(self):
        if self._recorder:
            self._recorder.stop()
        if self._server:
            self._server.shutdown()


# ── HTML PAGE ────────────────────────────────────────────────────────────
_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Algothon 2026 Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;color:#c9d1d9;padding:12px 16px}
  h1{color:#58a6ff;font-size:22px;margin-bottom:2px}
  .subtitle{color:#484f58;font-size:12px;margin-bottom:12px}
  h2{color:#8b949e;margin:18px 0 8px;font-size:14px;text-transform:uppercase;letter-spacing:1.5px;border-bottom:1px solid #21262d;padding-bottom:4px}
  .g2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  .g3{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
  .g4{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
  .g-auto{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px}
  .card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;overflow:hidden}
  .card-sm{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:8px 10px}
  .big{font-size:26px;font-weight:bold}
  .med{font-size:18px;font-weight:600}
  .green{color:#3fb950}.red{color:#f85149}.yellow{color:#d29922}.blue{color:#58a6ff}.dim{color:#484f58}.white{color:#e6edf3}
  table{width:100%;border-collapse:collapse;font-size:12px}
  th{text-align:left;color:#8b949e;border-bottom:1px solid #30363d;padding:3px 6px;font-weight:600}
  td{padding:3px 6px;border-bottom:1px solid #21262d}
  .badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:bold}
  .badge-ok{background:#0d3320;color:#3fb950}.badge-halt{background:#3d1416;color:#f85149}
  .bar-bg{height:5px;background:#30363d;border-radius:3px;margin-top:3px}
  .bar-fg{height:5px;border-radius:3px}
  .depth-row{display:flex;align-items:center;font-size:11px;height:18px;margin:1px 0}
  .depth-bar{height:16px;min-width:1px;border-radius:2px;opacity:0.7}
  .depth-price{width:55px;text-align:center;font-weight:600;flex-shrink:0}
  .depth-vol{width:28px;text-align:right;flex-shrink:0;color:#8b949e;font-size:10px}
  .depth-left{flex:1;display:flex;justify-content:flex-end;padding-right:4px}
  .depth-right{flex:1;display:flex;justify-content:flex-start;padding-left:4px}
  .pos-cell{font-weight:bold;font-size:13px}
  .tab-bar{display:flex;gap:4px;margin-bottom:6px;flex-wrap:wrap}
  .tab-btn{background:#21262d;color:#8b949e;border:1px solid #30363d;border-radius:4px;padding:3px 10px;cursor:pointer;font-size:11px}
  .tab-btn.active{background:#30363d;color:#58a6ff;border-color:#58a6ff}
  .chart-wrap{position:relative;height:200px}
  .chart-wrap canvas{position:absolute;top:0;left:0;width:100%!important;height:100%!important}
  #last-update{color:#484f58;font-size:11px}
  .pnl-api{font-size:11px;color:#484f58;margin-top:2px}
  .metric-row{display:flex;justify-content:space-between;padding:2px 0;font-size:12px}
  .metric-label{color:#8b949e}
  .pos-bar-wrap{display:flex;align-items:center;gap:4px}
  .pos-bar-track{width:80px;height:8px;background:#21262d;border-radius:4px;position:relative;overflow:hidden}
  .pos-bar-center{position:absolute;left:50%;top:0;width:1px;height:100%;background:#484f58}
  .pos-bar-fill{position:absolute;top:0;height:100%;border-radius:4px}
</style>
</head>
<body>

<div style="display:flex;justify-content:space-between;align-items:baseline">
  <div><h1>Algothon 2026 — Live Dashboard</h1><div class="subtitle">Imperial College | Challenge Exchange</div></div>
  <span id="last-update">Connecting...</span>
</div>

<!-- ═══ ROW 1: PnL + Status ═══ -->
<h2>Performance & Risk</h2>
<div class="g4" id="overview-cards"></div>

<!-- ═══ ROW 2: PnL chart (from API) ═══ -->
<h2>PnL History <span class="dim" style="font-size:11px">(from exchange API)</span></h2>
<div class="card"><div class="chart-wrap"><canvas id="pnl-chart"></canvas></div></div>

<!-- ═══ ROW 2b: Simulated PnL (dry-run only) ═══ -->
<div id="sim-section" style="display:none">
<h2>Simulated Strategy Performance <span class="dim" style="font-size:11px">(dry-run)</span></h2>
<div class="g4" id="sim-overview"></div>
<div class="g2" style="margin-top:10px">
  <div class="card"><div class="chart-wrap" style="height:220px"><canvas id="sim-pnl-chart"></canvas></div></div>
  <div>
    <div class="card" id="sim-strategy-table" style="max-height:220px;overflow-y:auto"></div>
  </div>
</div>
<div class="card" style="margin-top:10px;max-height:200px;overflow-y:auto">
  <div style="font-size:11px;color:#8b949e;margin-bottom:4px">Recent Simulated Trades</div>
  <table id="sim-trades-table">
    <thead><tr><th>Time</th><th>Strategy</th><th>Side</th><th>Product</th><th>Vol</th><th>Price</th><th>Realized</th><th>Pos After</th></tr></thead>
    <tbody></tbody>
  </table>
</div>
</div>

<!-- ═══ ROW 2a: Alpha Fair Values ═══ -->
<div id="alpha-section" style="display:none">
<h2>Alpha Fair Values <span class="dim" style="font-size:11px">(real-world data → settlement estimates)</span></h2>
<div class="g-auto" id="alpha-cards"></div>
</div>

<!-- ═══ ROW 3: Products table + Positions ═══ -->
<h2>Markets & Positions</h2>
<div class="card" style="overflow-x:auto">
  <table id="products-table">
    <thead><tr>
      <th>Product</th><th>Bid</th><th>Ask</th><th>Spread</th><th>Mid</th>
      <th>Fair</th><th>Conf</th>
      <th>EMA-F</th><th>EMA-S</th><th>Vol (bps)</th><th>Trend</th>
      <th>Imbalance</th><th>Position</th>
    </tr></thead>
    <tbody></tbody>
  </table>
</div>

<!-- ═══ ROW 4: Price Charts ═══ -->
<h2>Price Charts <button id="history-btn" onclick="loadFullHistory()" style="margin-left:12px;padding:3px 12px;background:#238636;color:#fff;border:1px solid #2ea043;border-radius:6px;cursor:pointer;font-size:12px">Load Full History</button> <span id="history-status" class="dim" style="font-size:11px"></span></h2>
<div class="card">
  <div class="tab-bar" id="price-tabs"></div>
  <div class="chart-wrap" style="height:300px"><canvas id="price-chart"></canvas></div>
</div>

<!-- ═══ ROW 5: Order Book Depth ═══ -->
<h2>Order Book Depth</h2>
<div class="g3" id="ob-section"></div>

<!-- ═══ ROW 6: ETF Arbitrage ═══ -->
<h2>ETF & FLY Arbitrage</h2>
<div class="g2">
  <div id="arb-cards" class="g2"></div>
  <div class="card"><div class="chart-wrap"><canvas id="arb-chart"></canvas></div></div>
</div>

<!-- ═══ ROW 7: Volatility & Advanced Metrics ═══ -->
<h2>Volatility & Metrics</h2>
<div class="g-auto" id="vol-section"></div>

<!-- ═══ ROW 8: Signal Stats ═══ -->
<h2>Signal & Rejection Stats</h2>
<div class="g-auto" id="signal-section"></div>

<!-- ═══ ROW 9: Market Volume ═══ -->
<h2>Market Volume <span class="dim" style="font-size:11px">(traded this session)</span></h2>
<div class="g-auto" id="volume-section"></div>

<script>
/* ── helpers ── */
const $=s=>document.querySelector(s);
const fmt=(v,d=2)=>v==null?'—':Number(v).toFixed(d);
const clr=v=>v==null?'dim':v>=0?'green':'red';
const COLORS=['#58a6ff','#3fb950','#f0883e','#bc8cff','#f85149','#d29922','#79c0ff','#ff7b72'];

/* ── Chart.js instances ── */
let pnlChart=null, priceChart=null, arbChart=null, simPnlChart=null;
let activePriceTab='ALL';
let fullHistoryData=null;
let historyMode=false;

function createLineChart(canvas, datasets, yLabel, xIsTime=true){
  const ctx=canvas.getContext('2d');
  return new Chart(ctx,{
    type:'line',
    data:{datasets},
    options:{
      responsive:true, maintainAspectRatio:false, animation:false,
      interaction:{mode:'index',intersect:false},
      scales:{
        x:{type:'linear',display:true,ticks:{callback:v=>{const d=new Date(v*1000);return d.getHours()+':'+String(d.getMinutes()).padStart(2,'0')},maxTicksLimit:8,color:'#484f58'},grid:{color:'#21262d'}},
        y:{display:true,ticks:{color:'#484f58',maxTicksLimit:6},grid:{color:'#21262d'}}
      },
      plugins:{legend:{display:datasets.length>1,position:'top',labels:{color:'#8b949e',boxWidth:10,font:{size:10}}},tooltip:{enabled:true}}
    }
  });
}

function updateLineChart(chart, datasets){
  if(!chart)return;
  chart.data.datasets=datasets;
  chart.update('none');
}

/* ── PnL chart ── */
function updatePnlChart(data){
  if(!data||!data.length)return;
  const ds=[{label:'PnL',data:data.map(d=>({x:d.t,y:d.pnl})),borderColor:'#3fb950',backgroundColor:'rgba(63,185,80,0.1)',fill:true,borderWidth:2,pointRadius:0,tension:0.3}];
  if(!pnlChart){pnlChart=createLineChart($('#pnl-chart'),ds,'PnL');}
  else updateLineChart(pnlChart,ds);
}

/* ── Simulated PnL (dry-run) ── */
function renderSimulator(sim){
  const section=$('#sim-section');
  if(!sim){section.style.display='none';return;}
  section.style.display='block';

  const totalPnl=sim.sim_total_pnl||0;
  const realPnl=sim.sim_realized_pnl||0;
  const unrealPnl=sim.sim_unrealized_pnl||0;
  const dd=sim.sim_drawdown||0;
  const trades=sim.sim_trade_count||0;
  const exposure=sim.sim_exposure||0;

  // Overview cards
  $('#sim-overview').innerHTML=`
    <div class="card">
      <div class="dim">Simulated Total PnL</div>
      <div class="big ${clr(totalPnl)}">${fmt(totalPnl,0)}</div>
      <div class="pnl-api">Realized: <span class="${clr(realPnl)}">${fmt(realPnl,0)}</span> | Unrealized: <span class="${clr(unrealPnl)}">${fmt(unrealPnl,0)}</span></div>
    </div>
    <div class="card">
      <div class="dim">Sim Drawdown</div>
      <div class="med ${clr(dd)}">${fmt(dd,0)}</div>
      <div class="pnl-api">Peak: ${fmt(sim.sim_peak_pnl,0)}</div>
    </div>
    <div class="card">
      <div class="dim">Sim Trades</div>
      <div class="med white">${trades}</div>
      <div class="pnl-api">Exposure: ${exposure}</div>
    </div>
    <div class="card">
      <div class="dim">Sim Positions</div>
      ${Object.entries(sim.sim_positions||{}).filter(([k,v])=>v!==0).map(([k,v])=>`<div class="metric-row"><span class="metric-label">${k}</span><span class="${clr(v)}">${v>0?'+':''}${v}</span></div>`).join('')||'<div class="dim" style="font-size:11px">Flat</div>'}
    </div>
  `;

  // PnL chart
  const pnlHist=sim.sim_pnl_history||[];
  if(pnlHist.length){
    const ds=[
      {label:'Sim Total PnL',data:pnlHist.map(d=>({x:d[0],y:d[1]})),borderColor:'#bc8cff',backgroundColor:'rgba(188,140,255,0.1)',fill:true,borderWidth:2,pointRadius:0,tension:0.3}
    ];
    if(!simPnlChart){simPnlChart=createLineChart($('#sim-pnl-chart'),ds,'Sim PnL');}
    else updateLineChart(simPnlChart,ds);
  }

  // Strategy table
  const spnl=sim.sim_strategy_pnl||{};
  const strades=sim.sim_strategy_trades||{};
  const svol=sim.sim_strategy_volume||{};
  const strats=Object.keys(spnl);
  let stratHtml='<div style="font-size:11px;color:#8b949e;margin-bottom:4px">Per-Strategy Attribution</div>';
  if(strats.length){
    stratHtml+='<table><thead><tr><th>Strategy</th><th>PnL</th><th>Trades</th><th>Volume</th></tr></thead><tbody>';
    strats.sort((a,b)=>(spnl[b]||0)-(spnl[a]||0));
    strats.forEach(s=>{
      stratHtml+=`<tr><td><strong class="white">${s.replace('dispatch_','')}</strong></td><td class="${clr(spnl[s])}">${fmt(spnl[s],0)}</td><td>${strades[s]||0}</td><td>${svol[s]||0}</td></tr>`;
    });
    stratHtml+='</tbody></table>';
  } else { stratHtml+='<div class="dim">No trades yet</div>'; }
  $('#sim-strategy-table').innerHTML=stratHtml;

  // Recent trades table
  const recent=sim.sim_recent_trades||[];
  const tbody=$('#sim-trades-table tbody');
  tbody.innerHTML=recent.slice().reverse().map(t=>{
    const d=new Date(t.time*1000);
    const ts=d.getHours()+':'+String(d.getMinutes()).padStart(2,'0')+':'+String(d.getSeconds()).padStart(2,'0');
    return `<tr>
      <td class="dim">${ts}</td>
      <td>${(t.strategy||'').replace('dispatch_','')}</td>
      <td class="${t.side==='BUY'?'green':'red'}">${t.side}</td>
      <td class="white">${t.product}</td>
      <td>${t.volume}</td>
      <td>${fmt(t.price,0)}</td>
      <td class="${clr(t.realized_pnl)}">${t.realized_pnl!==0?fmt(t.realized_pnl,0):'—'}</td>
      <td class="${clr(t.position_after)}">${t.position_after}</td>
    </tr>`;
  }).join('');
}

/* ── Price chart ── */
function updatePriceChart(history, tab){
  const products=Object.keys(history);
  if(!products.length)return;
  let ds=[];
  if(tab==='ALL'){
    products.forEach((sym,i)=>{
      const pts=history[sym]||[];
      if(!pts.length)return;
      // Normalize to percentage change for ALL view
      const base=pts[0].mid;
      ds.push({label:sym,data:pts.map(p=>({x:p.t,y:((p.mid-base)/base*100)})),borderColor:COLORS[i%COLORS.length],borderWidth:1.5,pointRadius:0,tension:0.3});
    });
  } else {
    const pts=history[tab]||[];
    if(pts.length){
      ds.push({label:tab+' Mid',data:pts.map(p=>({x:p.t,y:p.mid})),borderColor:'#58a6ff',borderWidth:2,pointRadius:0,tension:0.3});
      const bidPts=pts.filter(p=>p.bid!=null);
      if(bidPts.length) ds.push({label:'Bid',data:bidPts.map(p=>({x:p.t,y:p.bid})),borderColor:'rgba(63,185,80,0.4)',borderWidth:1,pointRadius:0,borderDash:[3,3]});
      const askPts=pts.filter(p=>p.ask!=null);
      if(askPts.length) ds.push({label:'Ask',data:askPts.map(p=>({x:p.t,y:p.ask})),borderColor:'rgba(248,81,73,0.4)',borderWidth:1,pointRadius:0,borderDash:[3,3]});
    }
  }
  if(!priceChart){priceChart=createLineChart($('#price-chart'),ds,'Price');}
  else updateLineChart(priceChart,ds);
}

/* ── Arb chart ── */
function updateArbChart(data){
  if(!data||!data.length)return;
  const ds=[
    {label:'ETF Mid',data:data.map(d=>({x:d.t,y:d.etf_mid})),borderColor:'#58a6ff',borderWidth:1.5,pointRadius:0,tension:0.3},
    {label:'Fair Value',data:data.map(d=>({x:d.t,y:d.fair})),borderColor:'#3fb950',borderWidth:1.5,pointRadius:0,tension:0.3,borderDash:[4,4]}
  ];
  if(!arbChart){arbChart=createLineChart($('#arb-chart'),ds,'ETF vs Fair');}
  else updateLineChart(arbChart,ds);
}

/* ── Order Book Depth Visualization ── */
function renderOrderBooks(products, orderbooks){
  const section=$('#ob-section');
  let html='';
  for(const p of products){
    const sym=p.product;
    const ob=orderbooks[sym];
    if(!ob)continue;
    const bids=(ob.bids||[]).slice(0,5);
    const asks=(ob.asks||[]).slice(0,5);
    const maxVol=Math.max(...bids.map(b=>b.vol),...asks.map(a=>a.vol),1);

    // Merge into ladder (asks reversed on top, bids below)
    let rows='';
    // Asks (reversed so best ask is at bottom)
    const asksRev=[...asks].reverse();
    for(const a of asksRev){
      const pct=(a.vol/maxVol*100).toFixed(0);
      rows+=`<div class="depth-row">
        <span class="depth-vol"></span>
        <div class="depth-left"></div>
        <span class="depth-price red">${fmt(a.price,0)}</span>
        <div class="depth-right"><div class="depth-bar" style="width:${pct}%;background:#f85149"></div></div>
        <span class="depth-vol">${a.vol}</span>
      </div>`;
    }
    // Spread line
    const sp=p.spread!=null?fmt(p.spread,1):'—';
    rows+=`<div class="depth-row" style="justify-content:center"><span class="dim" style="font-size:10px">spread: ${sp}</span></div>`;
    // Bids
    for(const b of bids){
      const pct=(b.vol/maxVol*100).toFixed(0);
      rows+=`<div class="depth-row">
        <span class="depth-vol">${b.vol}</span>
        <div class="depth-left"><div class="depth-bar" style="width:${pct}%;background:#3fb950"></div></div>
        <span class="depth-price green">${fmt(b.price,0)}</span>
        <div class="depth-right"></div>
        <span class="depth-vol"></span>
      </div>`;
    }

    html+=`<div class="card-sm">
      <div style="display:flex;justify-content:space-between;margin-bottom:4px">
        <strong class="white">${sym}</strong>
        <span class="dim" style="font-size:10px">${bids.length}b/${asks.length}a levels</span>
      </div>
      ${rows}
    </div>`;
  }
  section.innerHTML=html||'<div class="card"><div class="dim">No order book data</div></div>';
}

/* ── Position Bar ── */
function posBar(pos){
  const absPos=Math.abs(pos);
  const pct=(absPos/100*50);  // 100 is max, 50% is half the bar
  const left=pos>=0?50:50-pct;
  const c=pos>=0?'#3fb950':'#f85149';
  return `<div class="pos-bar-wrap">
    <span class="pos-cell ${clr(pos)}">${pos}</span>
    <div class="pos-bar-track">
      <div class="pos-bar-center"></div>
      <div class="pos-bar-fill" style="left:${left}%;width:${pct}%;background:${c}"></div>
    </div>
  </div>`;
}

/* ── RENDER ── */
function render(state){
  const r=state.risk||{};
  const pnlHist=state.pnl_history||[];
  const latestPnl=pnlHist.length?pnlHist[pnlHist.length-1].pnl:null;
  const dd=r.drawdown||0;
  const halted=r.halted;

  // -- Overview cards --
  $('#overview-cards').innerHTML=`
    <div class="card">
      <div class="dim">PnL (API)</div>
      <div class="big ${clr(latestPnl)}">${latestPnl!=null?fmt(latestPnl,0):'—'}</div>
      <div class="pnl-api">Local tracking: ${fmt(r.pnl,0)}</div>
    </div>
    <div class="card">
      <div class="dim">Status</div>
      <div class="med"><span class="badge ${halted?'badge-halt':'badge-ok'}">${halted?'HALTED':'RUNNING'}</span></div>
      ${halted?'<div class="dim" style="margin-top:4px">'+((r.halt_reason||''))+'</div>':''}
      ${(r.strategies_halted||[]).length?'<div class="dim" style="margin-top:2px;font-size:11px">Strats halted: '+(r.strategies_halted.join(', '))+'</div>':''}
    </div>
    <div class="card">
      <div class="dim">Signals</div>
      <div class="med">${r.signals_approved||0} <span class="dim" style="font-size:12px">/ ${r.signals_received||0}</span></div>
      <div class="dim" style="font-size:11px">Approval: ${fmt(r.approval_rate,1)}% | Rej: ${r.signals_rejected||0}</div>
    </div>
    <div class="card">
      <div class="dim">Risk</div>
      <div class="metric-row"><span class="metric-label">Peak PnL</span><span>${fmt(r.peak_pnl,0)}</span></div>
      <div class="metric-row"><span class="metric-label">Drawdown</span><span class="${clr(dd)}">${fmt(dd,0)}</span></div>
      <div class="metric-row"><span class="metric-label">Active Orders</span><span>${state.active_orders||0}</span></div>
      <div class="metric-row"><span class="metric-label">Data Files</span><span>${(state.data_files||[]).length}</span></div>
    </div>
  `;

  // -- PnL chart --
  updatePnlChart(pnlHist);

  // -- Simulated PnL (dry-run) --
  renderSimulator(state.simulator);

  // -- Alpha fair values --
  const alpha=state.alpha;
  const alphaSection=$('#alpha-section');
  if(alpha && alpha.fair && Object.keys(alpha.fair).length){
    alphaSection.style.display='block';
    const fairs=alpha.fair;
    const confs=alpha.confidence||{};
    let aHtml=`<div class="card-sm" style="min-width:180px">
      <div class="dim">Settlement In</div>
      <div class="big blue">${fmt(alpha.hours_left,1)}h</div>
      <div class="dim" style="font-size:11px">Last update: ${alpha.last_update?new Date(alpha.last_update*1000).toLocaleTimeString():'—'}</div>
    </div>`;
    const products=state.products||[];
    Object.entries(fairs).sort().forEach(([prod,fair])=>{
      if(fair==null)return;
      const conf=confs[prod]||0;
      const mp=products.find(p=>p.product===prod);
      const mid=mp?mp.mid:null;
      const edge=mid&&fair?((fair-mid)/mid*100):null;
      const edgeStr=edge!=null?`${edge>=0?'+':''}${edge.toFixed(1)}%`:'—';
      aHtml+=`<div class="card-sm">
        <div style="display:flex;justify-content:space-between"><strong class="white">${prod}</strong><span class="dim">${conf.toFixed(0)}% conf</span></div>
        <div class="metric-row"><span class="metric-label">Fair</span><span class="blue">${fmt(fair,0)}</span></div>
        <div class="metric-row"><span class="metric-label">Market</span><span>${mid!=null?fmt(mid,0):'—'}</span></div>
        <div class="metric-row"><span class="metric-label">Edge</span><span class="${edge>=0?'green':'red'}">${edgeStr}</span></div>
        <div class="bar-bg"><div class="bar-fg" style="width:${conf}%;background:#58a6ff"></div></div>
      </div>`;
    });
    $('#alpha-cards').innerHTML=aHtml;
  } else { alphaSection.style.display='none'; }

  // -- Products table (with alpha columns) --
  const tbody=$('#products-table tbody');
  tbody.innerHTML=(state.products||[]).map(p=>`
    <tr>
      <td><strong class="white">${p.product}</strong></td>
      <td class="green">${fmt(p.best_bid,0)}</td>
      <td class="red">${fmt(p.best_ask,0)}</td>
      <td>${fmt(p.spread,1)}</td>
      <td class="white">${fmt(p.mid,1)}</td>
      <td class="blue">${p.alpha_fair!=null?fmt(p.alpha_fair,0):'—'}</td>
      <td class="dim">${p.alpha_conf?p.alpha_conf+'%':'—'}</td>
      <td class="blue">${fmt(p.ema_fast,1)}</td>
      <td>${fmt(p.ema_slow,1)}</td>
      <td class="yellow">${fmt(p.volatility,1)}</td>
      <td class="${p.trend>=0?'green':'red'}">${fmt(p.trend,1)}</td>
      <td><span class="${p.imbalance>=0?'green':'red'}">${fmt(p.imbalance,3)}</span></td>
      <td>${posBar(p.position)}</td>
    </tr>
  `).join('');

  // -- Price tabs --
  const liveHistory=state.price_history||{};
  const priceHistory=historyMode&&fullHistoryData?fullHistoryData:liveHistory;
  const syms=Object.keys(priceHistory);
  const tabBar=$('#price-tabs');
  if(tabBar.children.length===0 || tabBar.dataset.syms!==syms.join(',')){
    tabBar.dataset.syms=syms.join(',');
    let tabHtml=`<button class="tab-btn ${activePriceTab==='ALL'?'active':''}" onclick="setTab('ALL')">ALL (%)</button>`;
    syms.forEach(s=>{tabHtml+=`<button class="tab-btn ${activePriceTab===s?'active':''}" onclick="setTab('${s}')">${s}</button>`;});
    tabBar.innerHTML=tabHtml;
  }
  updatePriceChart(priceHistory, activePriceTab);

  // -- Order books --
  renderOrderBooks(state.products||[], state.orderbooks||{});

  // -- Arb section --
  const arb=state.arb||{};
  $('#arb-cards').innerHTML=`
    <div class="card-sm">
      <div class="dim">LON_ETF vs Fair</div>
      <div class="med">${fmt(arb.etf_mid,0)} <span class="dim" style="font-size:12px">vs</span> <span class="green">${fmt(arb.fair_etf,0)}</span></div>
      <div class="metric-row"><span class="metric-label">Gap</span><span class="big ${clr(arb.gap)}">${fmt(arb.gap,0)}</span></div>
    </div>
    <div class="card-sm">
      <div class="dim">Components</div>
      <div class="metric-row"><span class="metric-label">TIDE_SPOT</span><span>${fmt(arb.tide_mid,0)}</span></div>
      <div class="metric-row"><span class="metric-label">WX_SPOT</span><span>${fmt(arb.wx_mid,0)}</span></div>
      <div class="metric-row"><span class="metric-label">LHR_COUNT</span><span>${fmt(arb.lhr_mid,0)}</span></div>
      <div class="metric-row" style="margin-top:4px;border-top:1px solid #21262d;padding-top:4px"><span class="metric-label">LON_FLY Mid</span><span>${fmt(arb.fly_mid,0)}</span></div>
      <div class="metric-row"><span class="metric-label">FLY Fair</span><span class="green">${fmt(arb.fly_fair,0)}</span></div>
    </div>
  `;
  updateArbChart(state.arb_history||[]);

  // -- Volatility & metrics --
  const volHtml=(state.products||[]).map(p=>`
    <div class="card-sm">
      <div style="display:flex;justify-content:space-between"><strong class="white">${p.product}</strong><span class="yellow">${fmt(p.volatility,1)} bps</span></div>
      <div class="metric-row"><span class="metric-label">EMA Fast</span><span class="blue">${fmt(p.ema_fast,1)}</span></div>
      <div class="metric-row"><span class="metric-label">EMA Slow</span><span>${fmt(p.ema_slow,1)}</span></div>
      <div class="metric-row"><span class="metric-label">Trend</span><span class="${p.trend>=0?'green':'red'}">${fmt(p.trend,1)} bps</span></div>
      <div class="metric-row"><span class="metric-label">Micro Price</span><span>${fmt(p.micro_price,1)}</span></div>
      <div class="metric-row"><span class="metric-label">Wtd Mid</span><span>${fmt(p.weighted_mid,1)}</span></div>
      <div class="metric-row"><span class="metric-label">Bid/Ask Vol</span><span>${p.bid_vol}/${p.ask_vol}</span></div>
      <div class="metric-row"><span class="metric-label">Levels B/A</span><span>${p.bid_levels}/${p.ask_levels}</span></div>
      <div class="metric-row"><span class="metric-label">Top Imb</span><span class="${p.top_imbalance>=0?'green':'red'}">${fmt(p.top_imbalance,3)}</span></div>
    </div>
  `).join('');
  $('#vol-section').innerHTML=volHtml||'<div class="card"><div class="dim">Waiting for data...</div></div>';

  // -- Signal rejection breakdown --
  const rej=r.rejection_breakdown||{};
  const rejTotal=Object.values(rej).reduce((a,b)=>a+b,0)||1;
  const rejHtml=Object.entries(rej).map(([k,v])=>`
    <div class="card-sm">
      <div style="display:flex;justify-content:space-between"><span class="dim">${k}</span><span class="yellow">${v}</span></div>
      <div class="bar-bg"><div class="bar-fg" style="width:${(v/rejTotal*100)}%;background:#d29922"></div></div>
    </div>
  `).join('');
  $('#signal-section').innerHTML=rejHtml||'<div class="card"><div class="dim">No rejections yet</div></div>';

  // -- Market Volume --
  const tvol=state.trade_volume||{};
  const tcnt=state.trade_count||{};
  const totalVol=Object.values(tvol).reduce((a,b)=>a+b,0);
  const totalCnt=Object.values(tcnt).reduce((a,b)=>a+b,0);
  const maxVol=Math.max(...Object.values(tvol),1);
  const volSorted=Object.entries(tvol).sort((a,b)=>b[1]-a[1]);
  let volSecHtml=`<div class="card-sm" style="min-width:200px">
    <div class="dim">Session Totals</div>
    <div class="big white">${totalVol.toLocaleString()} contracts</div>
    <div class="dim" style="font-size:11px">${totalCnt} trades across ${Object.keys(tvol).length} products</div>
  </div>`;
  volSorted.forEach(([prod,vol])=>{
    const cnt=tcnt[prod]||0;
    const pct=(vol/maxVol*100).toFixed(0);
    const low=vol<50;
    volSecHtml+=`<div class="card-sm">
      <div style="display:flex;justify-content:space-between"><strong class="${low?'red':'white'}">${prod}</strong><span class="${low?'red':'yellow'}">${vol.toLocaleString()}</span></div>
      <div class="bar-bg"><div class="bar-fg" style="width:${pct}%;background:${low?'#f85149':'#58a6ff'}"></div></div>
      <div class="dim" style="font-size:11px">${cnt} trades${low?' \u26a0\ufe0f LOW VOLUME':''}</div>
    </div>`;
  });
  $('#volume-section').innerHTML=volSecHtml||'<div class="card"><div class="dim">No trades yet</div></div>';
}

function setTab(tab){
  activePriceTab=tab;
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.toggle('active',b.textContent.trim().startsWith(tab)));
  // Force chart rebuild on tab switch
  if(priceChart){priceChart.destroy();priceChart=null;}
}

async function loadFullHistory(){
  const btn=$('#history-btn');
  const status=$('#history-status');
  if(historyMode){
    historyMode=false;
    btn.textContent='Load Full History';
    btn.style.background='#238636';
    status.textContent='';
    if(priceChart){priceChart.destroy();priceChart=null;}
    return;
  }
  btn.textContent='Loading...';
  btn.disabled=true;
  status.textContent='Fetching CSV data from disk...';
  try{
    const resp=await fetch('/api/history');
    const data=await resp.json();
    if(data.error){status.textContent='Error: '+data.error;btn.textContent='Load Full History';btn.disabled=false;return;}
    fullHistoryData=data.price_history||{};
    const totalPts=Object.values(fullHistoryData).reduce((a,b)=>a+b.length,0);
    const trades=(data.trade_history||[]).length;
    historyMode=true;
    btn.textContent='Switch to Live';
    btn.style.background='#8b949e';
    btn.disabled=false;
    status.textContent=`Showing ${totalPts.toLocaleString()} data points + ${trades} trades from disk`;
    if(priceChart){priceChart.destroy();priceChart=null;}
    // Force tab rebuild
    $('#price-tabs').innerHTML='';
    $('#price-tabs').dataset.syms='';
  }catch(e){
    status.textContent='Error: '+e.message;
    btn.textContent='Load Full History';
    btn.disabled=false;
  }
}

async function poll(){
  try{
    const resp=await fetch('/api/state');
    const data=await resp.json();
    render(data);
    const d=new Date(data.time*1000);
    $('#last-update').textContent='Updated '+d.toLocaleTimeString();
  }catch(e){
    $('#last-update').textContent='Error: '+e.message;
  }
}

setInterval(poll,2500);
poll();
</script>
</body>
</html>"""
