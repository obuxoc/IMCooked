# Algothon 2026 — Strategy Design Specification

> **Audience**: An AI assistant designing trading strategies for this project.  
> **Goal**: You will write Python functions that return `Signal | None`. You do NOT handle risk, execution, or data collection — those are handled by the framework.

---

## 1. YOUR JOB

Write a function with this exact signature:

```python
from signals import Signal, Side, OrderType

def my_strategy(bot: AlgothonBot, snap: OrderBookSnapshot) -> Signal | None:
    """Called on every SSE orderbook update for every product.

    Args:
        bot:  the live bot instance (access to all market data)
        snap: the latest OrderBookSnapshot for the product that just updated

    Returns:
        Signal  — if you want to trade
        None    — if you don't want to trade right now
    """
    if <your_condition>:
        return Signal(
            product=snap.product,
            side=Side.BUY,       # or Side.SELL
            price=snap.best_ask,  # or whatever price logic
            volume=5,
            order_type=OrderType.GTC,  # or IOC or QUOTE
            reason="my reason",        # for logging
        )
    return None
```

Register it in `run.py`:
```python
bot.register_orderbook_strategy(dispatch(my_strategy))
```

---

## 2. PRODUCTS

| Product      | Description                        | Notes                                    |
|--------------|-----------------------------------|------------------------------------------|
| `TIDE_SPOT`  | Tidal height spot                 | ETF component                            |
| `TIDE_SWING` | Tidal swing (range)               |                                          |
| `WX_SPOT`    | Weather spot                      | ETF component                            |
| `WX_SUM`     | Weather index (cumulative)        |                                          |
| `LHR_COUNT`  | Heathrow flights spot             | ETF component                            |
| `LHR_INDEX`  | Heathrow flights index            |                                          |
| `LON_ETF`    | Settles at `TIDE_SPOT + WX_SPOT + LHR_COUNT` | Arbitrage target           |
| `LON_FLY`    | Options butterfly on LON_ETF      | `2*Put(6200) + Call(6200) - 2*Call(6600) + 3*Call(7000)` |

---

## 3. OrderBookSnapshot — ALL AVAILABLE FIELDS

Your `snap` argument is an `OrderBookSnapshot` with these fields:

| Field              | Type    | Description                                                    |
|--------------------|---------|----------------------------------------------------------------|
| `snap.timestamp`   | `float` | Unix timestamp when SSE event was received                     |
| `snap.product`     | `str`   | Product symbol (e.g. `"TIDE_SPOT"`)                           |
| `snap.best_bid`    | `float` | Highest bid price (`NaN` if empty book)                       |
| `snap.best_ask`    | `float` | Lowest ask price (`NaN` if empty book)                        |
| `snap.mid`         | `float` | `(best_bid + best_ask) / 2`                                   |
| `snap.spread`      | `float` | `best_ask - best_bid`                                          |
| `snap.micro_price` | `float` | Volume-weighted mid: `(bid_vol*ask + ask_vol*bid) / (bid_vol+ask_vol)` |
| `snap.best_bid_vol`| `int`   | Volume at best bid                                             |
| `snap.best_ask_vol`| `int`   | Volume at best ask                                             |
| `snap.best_bid_own`| `int`   | OUR volume resting at best bid                                 |
| `snap.best_ask_own`| `int`   | OUR volume resting at best ask                                 |
| `snap.total_bid_vol`| `int`  | Sum of volume across ALL bid levels                            |
| `snap.total_ask_vol`| `int`  | Sum of volume across ALL ask levels                            |
| `snap.total_own_bid_vol` | `int` | Our total resting bid volume                              |
| `snap.total_own_ask_vol` | `int` | Our total resting ask volume                              |
| `snap.bid_levels`  | `int`   | Number of distinct bid price levels                            |
| `snap.ask_levels`  | `int`   | Number of distinct ask price levels                            |
| `snap.imbalance`   | `float` | `(total_bid - total_ask) / (total_bid + total_ask)` in [-1, 1] |
| `snap.top_imbalance` | `float` | Same but only at top-of-book                                |
| `snap.weighted_mid`| `float` | VWAP-style mid using top 3 levels each side                   |
| `snap.tick_size`   | `float` | Minimum price increment for this product                      |
| `snap.bid_prices_3`| `list[float]` | Top 3 bid prices `[best, 2nd, 3rd]`                   |
| `snap.bid_vols_3`  | `list[int]`   | Corresponding volumes                                  |
| `snap.ask_prices_3`| `list[float]` | Top 3 ask prices `[best, 2nd, 3rd]`                   |
| `snap.ask_vols_3`  | `list[int]`   | Corresponding volumes                                  |

---

## 4. BOT ACCESSORS — DATA YOU CAN READ

Access via the `bot` argument:

```python
# Latest snapshot for any product
snap_tide = bot.latest["TIDE_SPOT"]       # -> OrderBookSnapshot
snap_wx   = bot.latest["WX_SPOT"]

# All current mid prices
mids = bot.mids                           # -> {"TIDE_SPOT": 3500.0, "WX_SPOT": 2100.0, ...}

# Historical data as pandas DataFrame
df = bot.history("TIDE_SPOT", n=100)      # last 100 ticks
# df columns: timestamp, product, best_bid, best_ask, mid, spread, micro_price,
#             best_bid_vol, best_ask_vol, best_bid_own, best_ask_own,
#             total_bid_vol, total_ask_vol, total_own_bid_vol, total_own_ask_vol,
#             bid_levels, ask_levels, imbalance, top_imbalance, weighted_mid, tick_size

# Trade history
trades_df = bot.trade_history(n=50)       # last 50 trades
# trades_df columns: timestamp, exchange_ts, product, price, volume, buyer, seller

# ETF arbitrage helpers
fair_etf = bot.fair_etf                   # -> float: TIDE_SPOT + WX_SPOT + LHR_COUNT mids
arb_gap  = bot.etf_arb_gap              # -> float: LON_ETF mid - fair_etf (pos = overpriced)

# Full arb snapshot (dict)
arb = bot.cache.arb_snapshot()
# arb = {
#     "timestamp": 1234567890.0,
#     "etf_mid": 9100.0,
#     "fair_etf": 9050.0,
#     "gap": 50.0,
#     "tide_mid": 3500.0,
#     "wx_mid": 2100.0,
#     "lhr_mid": 3450.0,
#     "fly_mid": 500.0,
# }

# Multi-product correlation matrix
corr_df = bot.cache.multi_product_df(n=500)  # rows=timestamps, cols=product mids
```

---

## 5. SIGNAL — WHAT YOU RETURN

```python
from signals import Signal, Side, OrderType

Signal(
    product="TIDE_SPOT",        # REQUIRED: which product
    side=Side.BUY,              # REQUIRED: BUY or SELL
    price=3500.0,               # REQUIRED: desired price (auto-rounded to tick)
    volume=5,                   # REQUIRED: quantity (int, must be > 0)

    order_type=OrderType.GTC,   # OPTIONAL: GTC (default), IOC, or QUOTE
    reason="imbalance > 0.3",   # OPTIONAL: human-readable string for logs
    urgency=0.5,                # OPTIONAL: 0.0 (low) to 1.0 (high)

    # Only for QUOTE type (two-sided):
    ask_price=3510.0,           # OPTIONAL: ask side price
    ask_volume=5,               # OPTIONAL: ask side volume
)
```

### Order Types

| Type    | Behavior                                                                 |
|---------|-------------------------------------------------------------------------|
| `GTC`   | Rests on the book until filled or cancelled. For passive limit orders.  |
| `IOC`   | Immediately crosses the spread. Unfilled portion is cancelled.          |
| `QUOTE` | Places BOTH a bid and an ask. Requires `ask_price` and `ask_volume`.   |

### Side enum
```python
from signals import Side
Side.BUY   # "BUY"
Side.SELL  # "SELL"
```

---

## 6. WHAT THE RISK MANAGER CHECKS (you don't control this)

Your signal will be **rejected** if any of these fail:

| Check                    | Default Limit          | Description                              |
|--------------------------|------------------------|------------------------------------------|
| Trading halted           | —                      | Global or per-strategy halt active       |
| Order size               | max 20 per order       | `signal.volume <= 20`                    |
| Notional cap             | 100,000                | `price * volume <= 100,000`              |
| Position per product     | ±50                    | Projected abs position after fill        |
| Inventory hard limit     | ±40                    | Above this, only risk-reducing trades    |
| Total exposure           | 200                    | Sum of abs positions across all products |
| Drawdown                 | -2,000 from peak PnL   | Auto-halt when drawdown exceeds this     |
| PnL floor                | -5,000 total PnL       | Emergency halt                           |
| Rate limit               | 50 orders/minute       | Global                                   |
| Per-product throttle     | 1 signal/second        | Per product                              |
| Price                    | Must be > 0            |                                          |
| Quote spread             | bid < ask              | For QUOTE type only                      |
| Consecutive losses       | 10 per strategy        | Strategy halted; auto-resets after 120s  |

**Key insight**: If your position in a product reaches ±40, only SELL signals (if long) or BUY signals (if short) will be approved. Design strategies that naturally mean-revert.

---

## 7. EXCHANGE RULES

- **Rate limit**: Max 1 REST request per second (the framework handles this)
- **Order type**: Only GTC orders exist on the exchange. IOC is simulated (send + cancel).
- **Price-time priority**: Orders at same price fill in order of arrival.
- **Settlement**: Products settle at the end. LON_ETF = TIDE_SPOT + WX_SPOT + LHR_COUNT.
- **LON_FLY settlement**: `2*Put(6200) + Call(6200) - 2*Call(6600) + 3*Call(7000)` on LON_ETF final value.
- **No short-selling restrictions** — you can go long or short on any product.

---

## 8. STRATEGY IDEAS

### Market Making
Place bid and ask quotes around fair value. Profit from the spread. Use `imbalance` and `micro_price` to skew your quotes.

```python
def my_mm(bot, snap) -> Signal | None:
    if snap.spread < 3.0:
        return None  # spread too tight

    skew = snap.imbalance * 2.0  # positive imbalance = more buyers = skew ask up
    bid = snap.micro_price - snap.spread / 2 + skew
    ask = snap.micro_price + snap.spread / 2 + skew

    return Signal(
        product=snap.product,
        side=Side.BUY,  # side is for the bid leg
        price=bid,
        volume=3,
        order_type=OrderType.QUOTE,
        ask_price=ask,
        ask_volume=3,
        reason=f"MM spread={snap.spread:.1f} imb={snap.imbalance:.2f}",
    )
```

### ETF Arbitrage
When LON_ETF diverges from its fair value (sum of components), trade the gap.

```python
def my_arb(bot, snap) -> Signal | None:
    if snap.product != "LON_ETF":
        return None

    gap = bot.etf_arb_gap  # positive = ETF overpriced
    if abs(gap) < 10.0:
        return None  # gap too small

    side = Side.SELL if gap > 0 else Side.BUY
    price = snap.best_bid if side == Side.SELL else snap.best_ask

    return Signal(
        product="LON_ETF",
        side=side,
        price=price,
        volume=2,
        order_type=OrderType.IOC,
        reason=f"arb gap={gap:.1f}",
    )
```

### Momentum / Mean Reversion
Use `bot.history()` to detect trends or reversals.

```python
def my_momentum(bot, snap) -> Signal | None:
    df = bot.history(snap.product, n=50)
    if len(df) < 50:
        return None

    sma_fast = df["mid"].tail(10).mean()
    sma_slow = df["mid"].tail(50).mean()

    if sma_fast > sma_slow * 1.002:  # fast above slow by 0.2%
        return Signal(
            product=snap.product,
            side=Side.BUY,
            price=snap.best_ask,
            volume=3,
            order_type=OrderType.IOC,
            reason=f"momentum: fast={sma_fast:.0f} slow={sma_slow:.0f}",
        )
    return None
```

### Cross-Product Signals
Use one product's data to trade another.

```python
def cross_signal(bot, snap) -> Signal | None:
    if snap.product != "TIDE_SPOT":
        return None  # only trigger on TIDE_SPOT updates

    # If TIDE_SPOT imbalance is very positive, WX_SPOT might follow
    if snap.imbalance > 0.5:
        wx = bot.latest.get("WX_SPOT")
        if wx and wx.best_ask:
            return Signal(
                product="WX_SPOT",
                side=Side.BUY,
                price=wx.best_ask,
                volume=2,
                order_type=OrderType.IOC,
                reason=f"TIDE imbalance {snap.imbalance:.2f} -> buy WX",
            )
    return None
```

---

## 9. FILE STRUCTURE

```
algothon/
  algothon/
    bot_template.py       # Competition-provided base (DO NOT EDIT)
    data_cache.py         # OrderBookSnapshot + DataCache + DataPersistence
    algothon_bot.py       # AlgothonBot (extends BaseBot, wires SSE → cache)
    signals.py            # Signal, RiskManager, Executor, dispatcher
    strategies.py         # YOUR STRATEGY FUNCTIONS GO HERE
    dashboard.py          # Live web dashboard (http://localhost:8080)
    run.py                # Entry point — wire strategies here
    collected_data/       # Auto-saved CSVs (one per product + trades.csv)
```

---

## 10. HOW TO ADD A NEW STRATEGY

1. Write your function in `strategies.py` (or a new file):
   ```python
   def my_new_strategy(bot, snap) -> Signal | None:
       ...
   ```

2. Import and register it in `run.py`:
   ```python
   from strategies import my_new_strategy
   bot.register_orderbook_strategy(dispatch(my_new_strategy))
   ```

3. Run: `python run.py`

4. Open `http://localhost:8080` to see live performance.

---

## 11. DATA COLUMN REFERENCE (CSV FILES)

Each product CSV (`collected_data/TIDE_SPOT.csv`, etc.) has these columns:

```
timestamp, product, best_bid, best_ask, mid, spread, micro_price,
best_bid_vol, best_ask_vol, best_bid_own, best_ask_own,
total_bid_vol, total_ask_vol, total_own_bid_vol, total_own_ask_vol,
bid_levels, ask_levels, imbalance, top_imbalance, weighted_mid, tick_size
```

Trade CSV (`collected_data/trades.csv`):
```
timestamp, exchange_ts, product, price, volume, buyer, seller
```

---

## 12. IMPORTS CHEAT SHEET

```python
# Everything a strategy needs:
from signals import Signal, Side, OrderType
from data_cache import OrderBookSnapshot

# Type hint for the bot (optional):
from algothon_bot import AlgothonBot
```

---

## 13. CONSTRAINTS & TIPS

- **Return `None` often** — only signal when you're confident. The risk manager will reject marginal signals anyway.
- **Use `snap` fields directly** — they're pre-computed and fast. Don't re-parse the orderbook.
- **Avoid calling `bot.history()` on every tick** — it creates a DataFrame each time. Cache the result or use a counter.
- **Keep state in module-level variables** — your function is called per-tick. Use globals or closures for state.
- **`snap.product` tells you which product updated** — use `if snap.product != "TIDE_SPOT": return None` to filter.
- **The `bot.latest` dict has ALL products** — you can look at other products even when one updates.
- **Position awareness**: The risk manager tracks positions, but your strategy can also check `bot.get_positions()` (1 API call, rate-limited).
