"""Algothon 2026 -- Competition Dashboard v2.

Rebuilt from scratch for maximum competitive decision-making.

WHAT A TRADER NEEDS AT A GLANCE:
  1. PnL: total + per-product + trend
  2. Alpha edge: fair vs market, what is mispriced RIGHT NOW
  3. Positions: how close to limits, risk capacity
  4. Arbitrage: ETF gap, FLY gap (the money-makers)
  5. Risk: halted? rejection rate? signals throughput
  6. Price charts: clean, with fair value overlay
  7. Recent fills: what just happened

FIXED from v1:
  - NO API calls in recorder thread (was violating rate limit!)
  - PnL from API is pushed by run.py, not polled here
  - Full history uses vectorised pandas, not iterrows()
  - Clean single-purpose sections, no duplication
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

# -- Global refs (set by Dashboard.__init__) --
_bot: AlgothonBot | None = None
_risk: RiskManager | None = None
_executor: Executor | None = None
_persist: DataPersistence | None = None
_simulator = None
_alpha = None

# -- Rolling chart data --
_price_history: dict[str, deque] = {}
_arb_history: deque = deque(maxlen=500)
_pnl_history: deque = deque(maxlen=500)
_MAX_PTS = 300

# -- Trade tracking --
_trade_volume: dict[str, int] = {}
_trade_count: dict[str, int] = {}
_last_trade_idx: int = 0


def push_pnl(pnl_value: float) -> None:
    """Called by run.py when it fetches PnL -- avoids extra API calls."""
    _pnl_history.append({"t": time.time(), "pnl": round(float(pnl_value), 2)})


def _record_tick():
    """Snapshot prices for charts. NO API calls here (rate-limit safe)."""
    if _bot is None:
        return
    now = time.time()

    for sym, snap in _bot.latest.items():
        if sym not in _price_history:
            _price_history[sym] = deque(maxlen=_MAX_PTS)
        if not math.isnan(snap.mid):
            _price_history[sym].append({
                "t": now,
                "mid": round(snap.mid, 1),
                "bid": round(snap.best_bid, 1) if not math.isnan(snap.best_bid) else None,
                "ask": round(snap.best_ask, 1) if not math.isnan(snap.best_ask) else None,
            })

    # Arb gap
    try:
        arb = _bot.cache.arb_snapshot()
        gap = arb["gap"]
        if not math.isnan(gap):
            _arb_history.append({
                "t": now,
                "gap": round(gap, 1),
                "fair": round(arb["fair_etf"], 1),
                "etf_mid": round(arb["etf_mid"], 1),
            })
    except Exception:
        pass

    # Simulator MtM (no API)
    if _simulator and _bot:
        try:
            _simulator.mark_to_market(_bot.cache.mids)
        except Exception:
            pass

    # Trade volume counting
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


def _safe(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, float):
        return round(v, 2)
    return v


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/api/state":
            self._json(self._state())
        elif self.path == "/api/history":
            self._json(self._history())
        else:
            self._html()

    def _json(self, data):
        body = json.dumps(data, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(_HTML.encode())

    def _history(self) -> dict:
        """Full CSV history -- vectorised, fast."""
        result: dict = {"price_history": {}, "trade_history": []}
        if not _persist:
            return result
        try:
            import pandas as pd
            for product in _persist.get_all_products_on_disk():
                df = _persist.load_existing(product)
                if df.empty or "timestamp" not in df.columns:
                    continue
                df = df.sort_values("timestamp")
                cols = ["timestamp"]
                for c in ["mid", "best_bid", "best_ask"]:
                    if c in df.columns:
                        cols.append(c)
                sub = df[cols].copy()
                col_map = {"timestamp": "t", "mid": "mid",
                           "best_bid": "bid", "best_ask": "ask"}
                sub.columns = [col_map[c] for c in cols]
                for c in sub.columns[1:]:
                    sub[c] = sub[c].round(1)
                sub = sub.where(pd.notnull(sub), None)
                result["price_history"][product] = sub.to_dict("records")
        except Exception as e:
            result["error"] = str(e)
        return result

    def _state(self) -> dict:
        s: dict = {"time": time.time(), "products": [], "orderbooks": {}}
        if _bot is None:
            return s

        # Pre-fetch shared data
        tracker = None
        get_position = None
        compute_fly_fair_fn = None
        try:
            from strategies import (get_tracker, get_position as _gp,
                                    compute_fly_fair as _cff)
            tracker = get_tracker()
            get_position = _gp
            compute_fly_fair_fn = _cff
        except Exception:
            pass

        # -- Products --
        for sym, snap in _bot.latest.items():
            vol_val = ema_fast = ema_slow = trend = 0.0
            if tracker:
                try:
                    vol_val = tracker.volatility(sym)
                    ema_fast = tracker.ema_fast(sym)
                    ema_slow = tracker.ema_slow(sym)
                    trend = tracker.trend(sym)
                except Exception:
                    pass

            pos = 0
            if get_position:
                try:
                    pos = get_position(sym)
                except Exception:
                    pass

            alpha_fair = alpha_conf = None
            if _alpha:
                try:
                    afv, acf = _alpha.get(sym)
                    alpha_fair = _safe(afv)
                    alpha_conf = round(acf * 100, 1) if acf else 0
                except Exception:
                    pass

            entry = {
                "product": sym,
                "mid": _safe(snap.mid),
                "best_bid": _safe(snap.best_bid),
                "best_ask": _safe(snap.best_ask),
                "spread": _safe(snap.spread),
                "imbalance": round(snap.imbalance, 3),
                "bid_vol": snap.total_bid_vol,
                "ask_vol": snap.total_ask_vol,
                "position": pos,
                "volatility": round(vol_val * 10000, 2),
                "ema_fast": _safe(ema_fast),
                "ema_slow": _safe(ema_slow),
                "trend": round(trend * 10000, 1) if isinstance(trend, float) else 0,
                "alpha_fair": alpha_fair,
                "alpha_conf": alpha_conf or 0,
            }
            s["products"].append(entry)

            # Orderbook depth
            ob = {"bids": [], "asks": []}
            for p, v in zip(snap.bid_prices_3, snap.bid_vols_3):
                if not math.isnan(p):
                    ob["bids"].append({"price": round(p, 1), "vol": v})
            for p, v in zip(snap.ask_prices_3, snap.ask_vols_3):
                if not math.isnan(p):
                    ob["asks"].append({"price": round(p, 1), "vol": v})
            s["orderbooks"][sym] = ob

        # -- Risk --
        if _risk:
            rs = _risk.get_stats()
            s["risk"] = {
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
                "rejections": rs["rejection_breakdown"],
            }
        else:
            s["risk"] = {}

        # -- Charts --
        s["pnl_history"] = list(_pnl_history)
        s["price_history"] = {
            sym: list(h)[-_MAX_PTS:] for sym, h in _price_history.items()
        }
        s["arb_history"] = list(_arb_history)

        # -- Arb current --
        try:
            arb = _bot.cache.arb_snapshot()
            etf_ema = tracker.ema_slow("LON_ETF") if tracker else 0
            fly_fair = None
            if (compute_fly_fair_fn and etf_ema
                    and not math.isnan(etf_ema)):
                fly_fair = compute_fly_fair_fn(etf_ema)
            alpha_etf_fair = None
            if _alpha:
                try:
                    aef, _ = _alpha.get("LON_ETF")
                    alpha_etf_fair = _safe(aef)
                except Exception:
                    pass
            s["arb"] = {
                "etf_mid": _safe(arb["etf_mid"]),
                "fair_etf": _safe(arb["fair_etf"]),
                "alpha_etf_fair": alpha_etf_fair,
                "gap": _safe(arb["gap"]),
                "tide_mid": _safe(arb["tide_mid"]),
                "wx_mid": _safe(arb["wx_mid"]),
                "lhr_mid": _safe(arb["lhr_mid"]),
                "fly_mid": _safe(arb["fly_mid"]),
                "fly_fair": _safe(fly_fair),
            }
        except Exception:
            s["arb"] = {}

        # -- Executor --
        s["active_orders"] = (
            len(_executor._active_orders) if _executor else 0
        )

        # -- Simulator --
        if _simulator:
            ss = _simulator.get_stats()
            s["simulator"] = {
                "total_pnl": ss["sim_total_pnl"],
                "realized": ss["sim_realized_pnl"],
                "unrealized": ss["sim_unrealized_pnl"],
                "trade_count": ss["sim_trade_count"],
                "strategy_pnl": ss["sim_strategy_pnl"],
                "strategy_trades": ss["sim_strategy_trades"],
                "pnl_history": ss["sim_pnl_history"],
                "recent_trades": ss["sim_recent_trades"][-20:],
                "positions": ss["sim_positions"],
            }
        else:
            s["simulator"] = None

        # -- Alpha --
        if _alpha and _alpha.fair:
            s["alpha"] = {
                "fair": {k: _safe(v) for k, v in _alpha.fair.items()},
                "confidence": {
                    k: round(v * 100, 1) for k, v in _alpha.confidence.items()
                },
                "hours_left": round(_alpha._hours_left(), 2),
                "last_update": (
                    round(_alpha.last_update, 1) if _alpha.last_update
                    else None
                ),
            }
        else:
            s["alpha"] = None

        # -- Trade volume --
        s["trade_volume"] = dict(_trade_volume)
        s["trade_count"] = dict(_trade_count)

        return s


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
        self._server = None
        self._thread = None
        self._recorder = None

    def start(self):
        self._server = HTTPServer(("0.0.0.0", self.port), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()
        self._recorder = _Recorder(interval=3.0)
        self._recorder.start()
        print(f"[DASHBOARD] http://localhost:{self.port}")

    def stop(self):
        if self._recorder:
            self._recorder.stop()
        if self._server:
            self._server.shutdown()


# ===================================================================
# HTML / JS / CSS
# ===================================================================
_HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IMCooked - Algothon 2026</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#c9d1d9;--dim:#484f58;--green:#3fb950;--red:#f85149;--blue:#58a6ff;--yellow:#d29922;--purple:#bc8cff;--orange:#f0883e}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);font-size:13px;line-height:1.4}
.wrap{max-width:1600px;margin:0 auto;padding:10px 14px}
header{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid var(--border);margin-bottom:10px}
header h1{color:var(--blue);font-size:20px;font-weight:700}
header .meta{color:var(--dim);font-size:11px;text-align:right}
h2{color:var(--dim);font-size:12px;text-transform:uppercase;letter-spacing:1.2px;margin:14px 0 6px;font-weight:600}
.row{display:grid;gap:8px;margin-bottom:8px}
.r2{grid-template-columns:1fr 1fr}
.r3{grid-template-columns:1fr 1fr 1fr}
.r4{grid-template-columns:repeat(4,1fr)}
.r5{grid-template-columns:repeat(5,1fr)}
.r-auto{grid-template-columns:repeat(auto-fill,minmax(240px,1fr))}
.c{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:10px;overflow:hidden}
.c-tight{background:var(--card);border:1px solid var(--border);border-radius:5px;padding:6px 8px}
.hero{font-size:28px;font-weight:800}
.big{font-size:20px;font-weight:700}
.med{font-size:16px;font-weight:600}
.sm{font-size:11px}
.lbl{color:var(--dim);font-size:11px;margin-bottom:2px}
.grn{color:var(--green)}.red{color:var(--red)}.blu{color:var(--blue)}.ylw{color:var(--yellow)}.prp{color:var(--purple)}.dim{color:var(--dim)}.wht{color:#e6edf3}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700}
.badge-ok{background:#0d3320;color:var(--green)}.badge-halt{background:#3d1416;color:var(--red)}
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;color:var(--dim);border-bottom:1px solid var(--border);padding:4px 6px;font-weight:600;white-space:nowrap}
td{padding:4px 6px;border-bottom:1px solid #21262d;white-space:nowrap}
tr:hover{background:#1c2128}
.chart-w{position:relative;height:180px}
.chart-w canvas{position:absolute;inset:0;width:100%!important;height:100%!important}
.tab-bar{display:flex;gap:3px;margin-bottom:5px;flex-wrap:wrap}
.tab{background:#21262d;color:var(--dim);border:1px solid var(--border);border-radius:3px;padding:2px 8px;cursor:pointer;font-size:11px}
.tab.on{background:var(--border);color:var(--blue);border-color:var(--blue)}
.mr{display:flex;justify-content:space-between;padding:1px 0}
.bar-t{height:4px;background:#21262d;border-radius:2px;margin-top:2px;overflow:hidden}
.bar-f{height:4px;border-radius:2px}
.pos-wrap{display:flex;align-items:center;gap:4px}
.pos-track{width:70px;height:7px;background:#21262d;border-radius:3px;position:relative;overflow:hidden}
.pos-mid{position:absolute;left:50%;top:0;width:1px;height:100%;background:var(--dim)}
.pos-fill{position:absolute;top:0;height:100%;border-radius:3px}
.edge-pill{display:inline-block;padding:1px 6px;border-radius:3px;font-size:11px;font-weight:700}
.edge-pos{background:#0d3320;color:var(--green)}
.edge-neg{background:#3d1416;color:var(--red)}
.depth-row{display:flex;align-items:center;font-size:11px;height:16px;margin:1px 0}
.d-vol{width:24px;text-align:right;flex-shrink:0;color:var(--dim);font-size:10px}
.d-bar{height:14px;min-width:1px;border-radius:2px;opacity:0.6}
.d-price{width:50px;text-align:center;font-weight:600;flex-shrink:0;font-size:11px}
.d-left{flex:1;display:flex;justify-content:flex-end;padding-right:3px}
.d-right{flex:1;display:flex;justify-content:flex-start;padding-left:3px}
@media(max-width:900px){.r4,.r5{grid-template-columns:1fr 1fr}.r3{grid-template-columns:1fr}}
</style>
</head><body>
<div class="wrap">

<header>
  <div><h1>IMCooked - Algothon 2026</h1></div>
  <div class="meta"><span id="clock"></span><br><span id="settle-timer" class="ylw"></span></div>
</header>

<!-- 1. PnL + Status -->
<div class="row r5" id="hero"></div>

<!-- 2. Alpha Fair Values -->
<h2>Alpha Fair Values - Settlement Estimates</h2>
<div class="row r-auto" id="alpha-row"></div>

<!-- 3. Markets and Positions -->
<h2>Markets and Positions</h2>
<div class="c" style="overflow-x:auto"><table id="mkt-table">
  <thead><tr>
    <th>Product</th><th>Bid</th><th>Ask</th><th>Spread</th><th>Mid</th>
    <th>Fair Value</th><th>Edge</th><th>Conf</th>
    <th>Imbalance</th><th>Volume B/A</th><th>Position</th>
  </tr></thead>
  <tbody></tbody>
</table></div>

<!-- 4. Charts: PnL + Prices -->
<h2>Charts</h2>
<div class="row r2">
  <div class="c">
    <div class="lbl">PnL Over Time</div>
    <div class="chart-w"><canvas id="ch-pnl"></canvas></div>
  </div>
  <div class="c">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
      <div class="lbl">Prices</div>
      <div>
        <button id="hist-btn" onclick="toggleHistory()" class="tab" style="font-size:10px">Load History</button>
        <span id="hist-status" class="dim sm"></span>
      </div>
    </div>
    <div class="tab-bar" id="p-tabs"></div>
    <div class="chart-w" style="height:200px"><canvas id="ch-price"></canvas></div>
  </div>
</div>

<!-- 5. ETF and FLY Arbitrage -->
<h2>Arbitrage</h2>
<div class="row r3">
  <div class="c" id="arb-etf"></div>
  <div class="c" id="arb-fly"></div>
  <div class="c"><div class="lbl">ETF Gap History</div><div class="chart-w"><canvas id="ch-arb"></canvas></div></div>
</div>

<!-- 6. Order Book Depth -->
<h2>Order Books</h2>
<div class="row r-auto" id="ob-row"></div>

<!-- 7. Strategy Performance + Recent Trades -->
<h2>Strategy Performance</h2>
<div class="row r2">
  <div class="c" id="strat-table"></div>
  <div class="c" style="max-height:260px;overflow-y:auto">
    <div class="lbl">Recent Fills</div>
    <table id="trade-log"><thead><tr><th>Time</th><th>Strategy</th><th>Side</th><th>Product</th><th>Vol</th><th>Price</th><th>PnL</th></tr></thead><tbody></tbody></table>
  </div>
</div>

<!-- 8. Risk and Signals -->
<h2>Risk and Signals</h2>
<div class="row r-auto" id="risk-row"></div>

</div>

<script>
var $sel=function(s){return document.querySelector(s)};
var fmt=function(v,d){d=d||1;return v==null||v===undefined?'\u2014':Number(v).toFixed(d)};
var clr=function(v){return v==null?'dim':v>=0?'grn':'red'};
var COLORS=['#58a6ff','#3fb950','#f0883e','#bc8cff','#f85149','#d29922','#79c0ff','#ff7b72'];

var chPnl=null, chPrice=null, chArb=null;
var activeTab='ALL', histData=null, histMode=false;

function mkChart(el, datasets, opts){
  opts=opts||{};
  return new Chart(el.getContext('2d'),{
    type:'line', data:{datasets:datasets},
    options:{
      responsive:true, maintainAspectRatio:false, animation:false,
      interaction:{mode:'index',intersect:false},
      scales:{
        x:{type:'linear',display:true,ticks:{callback:function(v){var d=new Date(v*1000);return d.getHours()+':'+String(d.getMinutes()).padStart(2,'0')},maxTicksLimit:6,color:'#484f58'},grid:{color:'#1c2128'}},
        y:{display:true,ticks:{color:'#484f58',maxTicksLimit:5},grid:{color:'#1c2128'}}
      },
      plugins:{legend:{display:datasets.length>1,position:'top',labels:{color:'#8b949e',boxWidth:8,font:{size:10}}},tooltip:{enabled:true}}
    }
  });
}

function updChart(ch, ds){if(ch){ch.data.datasets=ds;ch.update('none');}}

function posBar(pos){
  var a=Math.abs(pos), p=(a/100*50);
  var l=pos>=0?50:50-p;
  var c=pos>=0?'var(--green)':'var(--red)';
  var warn=a>=60?' ylw':'';
  return '<div class="pos-wrap">'+
    '<strong class="'+clr(pos)+warn+'" style="width:30px;text-align:right">'+pos+'</strong>'+
    '<div class="pos-track"><div class="pos-mid"></div><div class="pos-fill" style="left:'+l+'%;width:'+p+'%;background:'+c+'"></div></div>'+
  '</div>';
}

function edgePill(mid, fair){
  if(mid==null||fair==null||mid===0) return '<span class="dim">\u2014</span>';
  var e=(fair-mid)/mid*100;
  var cls=e>=0?'edge-pos':'edge-neg';
  return '<span class="edge-pill '+cls+'">'+(e>=0?'+':'')+e.toFixed(1)+'%</span>';
}

function render(S){
  var r=S.risk||{};
  var prods=S.products||[];
  var sim=S.simulator;
  var alpha=S.alpha;
  var arb=S.arb||{};

  // Clock + settlement timer
  var now=new Date(S.time*1000);
  $sel('#clock').textContent='Updated '+now.toLocaleTimeString();
  if(alpha&&alpha.hours_left!=null){
    var h=alpha.hours_left;
    $sel('#settle-timer').textContent=h>0?'Settlement in '+h.toFixed(1)+'h':'Settlement passed';
  }

  // 1. Hero cards
  var pnlH=S.pnl_history||[];
  var curPnl=pnlH.length?pnlH[pnlH.length-1].pnl:(r.pnl||null);
  var prevPnl=pnlH.length>10?pnlH[pnlH.length-11].pnl:null;
  var pnlDelta=(curPnl!=null&&prevPnl!=null)?curPnl-prevPnl:null;
  var totalPos=prods.reduce(function(a,p){return a+Math.abs(p.position||0)},0);
  var maxExposure=800;
  var expPct=(totalPos/maxExposure*100);
  var simPnl=sim?sim.total_pnl:null;
  var halted=r.halted;

  var heroHtml='<div class="c">'+
    '<div class="lbl">PnL (Exchange)</div>'+
    '<div class="hero '+clr(curPnl)+'">'+(curPnl!=null?Math.round(curPnl).toLocaleString():'\u2014')+'</div>';
  if(pnlDelta!=null) heroHtml+='<div class="sm '+clr(pnlDelta)+'">delta30s: '+(pnlDelta>=0?'+':'')+Math.round(pnlDelta).toLocaleString()+'</div>';
  heroHtml+='</div>';

  heroHtml+='<div class="c">'+
    '<div class="lbl">Peak / Drawdown</div>'+
    '<div class="big wht">'+fmt(r.peak_pnl,0)+'</div>'+
    '<div class="sm '+clr(r.drawdown)+'">DD: '+fmt(r.drawdown,0)+'</div>'+
  '</div>';

  heroHtml+='<div class="c">'+
    '<div class="lbl">Exposure</div>'+
    '<div class="big '+(expPct>90?'red':expPct>70?'ylw':'wht')+'">'+totalPos+' / '+maxExposure+'</div>'+
    '<div class="bar-t"><div class="bar-f" style="width:'+expPct+'%;background:'+(expPct>80?'var(--red)':expPct>50?'var(--yellow)':'var(--green)')+'"></div></div>'+
  '</div>';

  heroHtml+='<div class="c">'+
    '<div class="lbl">Signals</div>'+
    '<div class="big wht">'+(r.signals_approved||0)+'<span class="dim sm"> / '+(r.signals_received||0)+'</span></div>'+
    '<div class="sm dim">Rate: '+fmt(r.approval_rate,0)+'% | Orders: '+(S.active_orders||0)+'</div>'+
  '</div>';

  heroHtml+='<div class="c">'+
    '<div class="lbl">Status</div>'+
    '<div class="med"><span class="badge '+(halted?'badge-halt':'badge-ok')+'">'+(halted?'HALTED':'LIVE')+'</span></div>';
  if(halted) heroHtml+='<div class="sm red">'+(r.halt_reason||'')+'</div>';
  if(sim) heroHtml+='<div class="sm prp" style="margin-top:4px">Sim PnL: '+fmt(simPnl,0)+'</div>';
  heroHtml+='</div>';

  $sel('#hero').innerHTML=heroHtml;

  // 2. Alpha fair values
  if(alpha&&alpha.fair){
    var fairs=alpha.fair;
    var confs=alpha.confidence||{};
    var ah='';
    Object.keys(fairs).sort().forEach(function(prod){
      var fair=fairs[prod];
      if(fair==null)return;
      var conf=confs[prod]||0;
      var mp=prods.find(function(p){return p.product===prod});
      var mid=mp?mp.mid:null;
      ah+='<div class="c-tight">'+
        '<div style="display:flex;justify-content:space-between"><strong class="wht">'+prod+'</strong>'+edgePill(mid,fair)+'</div>'+
        '<div class="mr"><span class="dim">Fair</span><span class="blu">'+fmt(fair,0)+'</span></div>'+
        '<div class="mr"><span class="dim">Market</span><span>'+(mid!=null?fmt(mid,0):'\u2014')+'</span></div>'+
        '<div class="bar-t"><div class="bar-f" style="width:'+conf+'%;background:var(--blue)"></div></div>'+
        '<div class="sm dim" style="text-align:right">'+conf.toFixed(0)+'% confidence</div>'+
      '</div>';
    });
    if(!ah) ah='<div class="c-tight"><div class="dim">Alpha engine warming up...</div></div>';
    $sel('#alpha-row').innerHTML=ah;
  } else {
    $sel('#alpha-row').innerHTML='<div class="c-tight"><div class="dim">Alpha engine not connected</div></div>';
  }

  // 3. Markets table
  var tb=$sel('#mkt-table tbody');
  tb.innerHTML=prods.map(function(p){
    var edge=edgePill(p.mid, p.alpha_fair);
    return '<tr>'+
      '<td><strong class="wht">'+p.product+'</strong></td>'+
      '<td class="grn">'+fmt(p.best_bid,0)+'</td>'+
      '<td class="red">'+fmt(p.best_ask,0)+'</td>'+
      '<td>'+fmt(p.spread,1)+'</td>'+
      '<td class="wht">'+fmt(p.mid,0)+'</td>'+
      '<td class="blu">'+(p.alpha_fair!=null?fmt(p.alpha_fair,0):'\u2014')+'</td>'+
      '<td>'+edge+'</td>'+
      '<td class="dim">'+(p.alpha_conf?p.alpha_conf.toFixed(0)+'%':'\u2014')+'</td>'+
      '<td><span class="'+(p.imbalance>=0?'grn':'red')+'">'+fmt(p.imbalance,3)+'</span></td>'+
      '<td class="dim">'+p.bid_vol+'/'+p.ask_vol+'</td>'+
      '<td>'+posBar(p.position)+'</td>'+
    '</tr>';
  }).join('');

  // 4a. PnL chart
  if(pnlH.length){
    var pnlDs=[{label:'PnL',data:pnlH.map(function(d){return{x:d.t,y:d.pnl}}),borderColor:'#3fb950',backgroundColor:'rgba(63,185,80,0.08)',fill:true,borderWidth:2,pointRadius:0,tension:0.3}];
    if(sim&&sim.pnl_history&&sim.pnl_history.length){
      pnlDs.push({label:'Sim PnL',data:sim.pnl_history.map(function(d){return{x:d[0],y:d[1]}}),borderColor:'#bc8cff',borderWidth:1.5,pointRadius:0,tension:0.3,borderDash:[4,4]});
    }
    if(!chPnl) chPnl=mkChart($sel('#ch-pnl'),pnlDs);
    else updChart(chPnl,pnlDs);
  }

  // 4b. Price chart
  var liveH=S.price_history||{};
  var pH=histMode&&histData?histData:liveH;
  var syms=Object.keys(pH);
  var tb2=$sel('#p-tabs');
  if(tb2.children.length===0||tb2.dataset.k!==syms.join()){
    tb2.dataset.k=syms.join();
    var tabsHtml='<div class="tab '+(activeTab==='ALL'?'on':'')+'" onclick="setTab(\'ALL\')">ALL %</div>';
    syms.forEach(function(s){
      tabsHtml+='<div class="tab '+(activeTab===s?'on':'')+'" onclick="setTab(\''+s+'\')">'+s+'</div>';
    });
    tb2.innerHTML=tabsHtml;
  }
  if(syms.length){
    var ds=[];
    if(activeTab==='ALL'){
      syms.forEach(function(s,i){
        var pts=pH[s]||[];
        if(!pts.length)return;
        var base=pts[0].mid;
        if(!base)return;
        ds.push({label:s,data:pts.filter(function(p){return p.mid!=null}).map(function(p){return{x:p.t,y:((p.mid-base)/base*100)}}),borderColor:COLORS[i%8],borderWidth:1.5,pointRadius:0,tension:0.3});
      });
    } else {
      var pts=pH[activeTab]||[];
      if(pts.length){
        ds.push({label:activeTab,data:pts.filter(function(p){return p.mid!=null}).map(function(p){return{x:p.t,y:p.mid}}),borderColor:'#58a6ff',borderWidth:2,pointRadius:0,tension:0.3});
        var ap=prods.find(function(p){return p.product===activeTab});
        if(ap&&ap.alpha_fair!=null&&pts.length>1){
          var fv=ap.alpha_fair;
          ds.push({label:'Fair (alpha)',data:[{x:pts[0].t,y:fv},{x:pts[pts.length-1].t,y:fv}],borderColor:'#d29922',borderWidth:1.5,pointRadius:0,borderDash:[6,4]});
        }
        var bidPts=pts.filter(function(p){return p.bid!=null});
        if(bidPts.length) ds.push({label:'Bid',data:bidPts.map(function(p){return{x:p.t,y:p.bid}}),borderColor:'rgba(63,185,80,0.3)',borderWidth:1,pointRadius:0});
        var askPts=pts.filter(function(p){return p.ask!=null});
        if(askPts.length) ds.push({label:'Ask',data:askPts.map(function(p){return{x:p.t,y:p.ask}}),borderColor:'rgba(248,81,73,0.3)',borderWidth:1,pointRadius:0});
      }
    }
    if(!chPrice) chPrice=mkChart($sel('#ch-price'),ds);
    else updChart(chPrice,ds);
  }

  // 5. Arbitrage
  var alphaETF=arb.alpha_etf_fair;
  var etfHtml='<div class="lbl">LON_ETF Arbitrage</div>'+
    '<div class="mr"><span class="dim">ETF Market</span><span class="wht big">'+fmt(arb.etf_mid,0)+'</span></div>'+
    '<div class="mr"><span class="dim">Component Fair</span><span class="grn">'+fmt(arb.fair_etf,0)+'</span></div>';
  if(alphaETF) etfHtml+='<div class="mr"><span class="dim">Alpha Fair</span><span class="blu">'+fmt(alphaETF,0)+'</span></div>';
  etfHtml+='<div class="mr" style="margin-top:4px;padding-top:4px;border-top:1px solid var(--border)"><span class="dim">Gap (Mid-Fair)</span><span class="big '+clr(arb.gap)+'">'+fmt(arb.gap,0)+'</span></div>'+
    '<div style="margin-top:6px">'+
    '<div class="mr sm"><span class="dim">TIDE_SPOT</span><span>'+fmt(arb.tide_mid,0)+'</span></div>'+
    '<div class="mr sm"><span class="dim">WX_SPOT</span><span>'+fmt(arb.wx_mid,0)+'</span></div>'+
    '<div class="mr sm"><span class="dim">LHR_COUNT</span><span>'+fmt(arb.lhr_mid,0)+'</span></div>'+
  '</div>';
  $sel('#arb-etf').innerHTML=etfHtml;

  var flyHtml='<div class="lbl">LON_FLY Arbitrage</div>'+
    '<div class="mr"><span class="dim">FLY Market</span><span class="wht big">'+fmt(arb.fly_mid,0)+'</span></div>'+
    '<div class="mr"><span class="dim">FLY Fair</span><span class="grn">'+fmt(arb.fly_fair,0)+'</span></div>';
  if(arb.fly_mid!=null&&arb.fly_fair!=null){
    flyHtml+='<div class="mr" style="margin-top:4px;padding-top:4px;border-top:1px solid var(--border)"><span class="dim">Gap</span><span class="big '+clr(arb.fly_mid-arb.fly_fair)+'">'+fmt(arb.fly_mid-arb.fly_fair,0)+'</span></div>';
  }
  flyHtml+='<div class="sm dim" style="margin-top:6px">2xPut(6200) + Call(6200) - 2xCall(6600) + 3xCall(7000)</div>';
  $sel('#arb-fly').innerHTML=flyHtml;

  // Arb chart
  var arbH=S.arb_history||[];
  if(arbH.length){
    var arbDs=[
      {label:'ETF Mid',data:arbH.map(function(d){return{x:d.t,y:d.etf_mid}}),borderColor:'#58a6ff',borderWidth:1.5,pointRadius:0,tension:0.3},
      {label:'Fair Value',data:arbH.map(function(d){return{x:d.t,y:d.fair}}),borderColor:'#3fb950',borderWidth:1.5,pointRadius:0,tension:0.3,borderDash:[4,4]}
    ];
    if(!chArb) chArb=mkChart($sel('#ch-arb'),arbDs);
    else updChart(chArb,arbDs);
  }

  // 6. Order books
  var obs=S.orderbooks||{};
  var obHtml='';
  prods.forEach(function(p){
    var sym=p.product;
    var ob=obs[sym];
    if(!ob)return;
    var bids=ob.bids||[];
    var asks=ob.asks||[];
    var allVols=bids.map(function(b){return b.vol}).concat(asks.map(function(a){return a.vol}));
    var maxV=Math.max.apply(null,allVols.concat([1]));
    var rows='';
    asks.slice().reverse().forEach(function(a){
      var w=(a.vol/maxV*100).toFixed(0);
      rows+='<div class="depth-row"><span class="d-vol"></span><div class="d-left"></div><span class="d-price red">'+fmt(a.price,0)+'</span><div class="d-right"><div class="d-bar" style="width:'+w+'%;background:var(--red)"></div></div><span class="d-vol">'+a.vol+'</span></div>';
    });
    rows+='<div class="depth-row" style="justify-content:center"><span class="dim" style="font-size:10px">sprd: '+fmt(p.spread,0)+'</span></div>';
    bids.forEach(function(b){
      var w=(b.vol/maxV*100).toFixed(0);
      rows+='<div class="depth-row"><span class="d-vol">'+b.vol+'</span><div class="d-left"><div class="d-bar" style="width:'+w+'%;background:var(--green)"></div></div><span class="d-price grn">'+fmt(b.price,0)+'</span><div class="d-right"></div><span class="d-vol"></span></div>';
    });
    obHtml+='<div class="c-tight"><div style="display:flex;justify-content:space-between;margin-bottom:3px"><strong class="wht">'+sym+'</strong><span class="dim sm">'+posBar(p.position)+'</span></div>'+rows+'</div>';
  });
  $sel('#ob-row').innerHTML=obHtml||'<div class="c-tight dim">No data</div>';

  // 7. Strategy table + trade log
  if(sim){
    var sp=sim.strategy_pnl||{};
    var st=sim.strategy_trades||{};
    var strats=Object.keys(sp).sort(function(a,b){return(sp[b]||0)-(sp[a]||0)});
    var sh='<div class="lbl">Per-Strategy Attribution (Simulated)</div><table><thead><tr><th>Strategy</th><th>PnL</th><th>Trades</th><th>PnL/Trade</th></tr></thead><tbody>';
    strats.forEach(function(s){
      var pnl=sp[s]||0;
      var trades=st[s]||0;
      var avg=trades?pnl/trades:0;
      var name=s.replace('dispatch_','');
      sh+='<tr><td class="wht">'+name+'</td><td class="'+clr(pnl)+'">'+fmt(pnl,0)+'</td><td>'+trades+'</td><td class="'+clr(avg)+'">'+fmt(avg,0)+'</td></tr>';
    });
    sh+='</tbody></table>';
    $sel('#strat-table').innerHTML=sh;

    var recent=sim.recent_trades||[];
    var tlb=$sel('#trade-log tbody');
    tlb.innerHTML=recent.slice().reverse().map(function(t){
      var d=new Date(t.time*1000);
      var ts=d.getHours()+':'+String(d.getMinutes()).padStart(2,'0')+':'+String(d.getSeconds()).padStart(2,'0');
      return '<tr><td class="dim">'+ts+'</td><td class="sm">'+(t.strategy||'').replace('dispatch_','')+'</td><td class="'+(t.side==='BUY'?'grn':'red')+'">'+t.side+'</td><td class="wht">'+t.product+'</td><td>'+t.volume+'</td><td>'+fmt(t.price,0)+'</td><td class="'+clr(t.realized_pnl)+'">'+(t.realized_pnl?fmt(t.realized_pnl,0):'\u2014')+'</td></tr>';
    }).join('');
  } else {
    $sel('#strat-table').innerHTML='<div class="dim sm">Strategy data available after first trades</div>';
  }

  // 8. Risk and rejections
  var rej=r.rejections||{};
  var rejKeys=Object.keys(rej);
  var rejTotal=Object.values(rej).reduce(function(a,b){return a+b},0)||1;
  var rh='<div class="c-tight">'+
    '<div class="lbl">Risk Summary</div>'+
    '<div class="mr"><span class="dim">PnL (local)</span><span class="'+clr(r.pnl)+'">'+fmt(r.pnl,0)+'</span></div>'+
    '<div class="mr"><span class="dim">Peak</span><span>'+fmt(r.peak_pnl,0)+'</span></div>'+
    '<div class="mr"><span class="dim">Drawdown</span><span class="'+clr(r.drawdown)+'">'+fmt(r.drawdown,0)+'</span></div>'+
    '<div class="mr"><span class="dim">Active Orders</span><span>'+(S.active_orders||0)+'</span></div>';
  if(r.strategies_halted&&r.strategies_halted.length) rh+='<div class="mr"><span class="dim">Halted Strats</span><span class="red">'+r.strategies_halted.join(", ")+'</span></div>';
  rh+='</div>';
  if(rejKeys.length){
    var rejTotalV=Object.values(rej).reduce(function(a,b){return a+b},0);
    rh+='<div class="c-tight"><div class="lbl">Rejection Breakdown ('+rejTotalV+' total)</div>';
    rejKeys.sort(function(a,b){return rej[b]-rej[a]}).forEach(function(k){
      rh+='<div class="mr"><span class="dim">'+k+'</span><span class="ylw">'+rej[k]+'</span></div><div class="bar-t"><div class="bar-f" style="width:'+(rej[k]/rejTotal*100)+'%;background:var(--yellow)"></div></div>';
    });
    rh+='</div>';
  }
  var tvol=S.trade_volume||{};
  var totalVol=Object.values(tvol).reduce(function(a,b){return a+b},0);
  if(totalVol){
    rh+='<div class="c-tight"><div class="lbl">Session Volume: '+totalVol.toLocaleString()+' contracts</div>';
    Object.entries(tvol).sort(function(a,b){return b[1]-a[1]}).forEach(function(e){
      rh+='<div class="mr sm"><span class="dim">'+e[0]+'</span><span>'+e[1].toLocaleString()+'</span></div>';
    });
    rh+='</div>';
  }
  $sel('#risk-row').innerHTML=rh;
}

function setTab(t){
  activeTab=t;
  document.querySelectorAll('#p-tabs .tab').forEach(function(b){b.classList.toggle('on',b.textContent.trim()===t||(b.textContent.trim()==='ALL %'&&t==='ALL'))});
  if(chPrice){chPrice.destroy();chPrice=null;}
}

function toggleHistory(){
  var btn=$sel('#hist-btn'), st=$sel('#hist-status');
  if(histMode){histMode=false;btn.textContent='Load History';st.textContent='';if(chPrice){chPrice.destroy();chPrice=null;$sel('#p-tabs').innerHTML='';$sel('#p-tabs').dataset.k='';}return;}
  btn.textContent='...';btn.disabled=true;st.textContent='Loading CSVs...';
  fetch('/api/history').then(function(r){return r.json()}).then(function(d){
    if(d.error){st.textContent='Error: '+d.error;btn.textContent='Load History';btn.disabled=false;return;}
    histData=d.price_history||{};
    var pts=Object.values(histData).reduce(function(a,b){return a+b.length},0);
    histMode=true;btn.textContent='Live Mode';btn.disabled=false;
    st.textContent=pts.toLocaleString()+' points loaded';
    if(chPrice){chPrice.destroy();chPrice=null;$sel('#p-tabs').innerHTML='';$sel('#p-tabs').dataset.k='';}
  }).catch(function(e){st.textContent='Error: '+e.message;btn.textContent='Load History';btn.disabled=false;});
}

function poll(){
  fetch('/api/state').then(function(r){return r.json()}).then(function(d){render(d)}).catch(function(e){$sel('#clock').textContent='Error: '+e.message});
}

setInterval(poll,2500);
poll();
</script>
</body>
</html>"""
