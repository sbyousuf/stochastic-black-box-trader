"""
STOCHASTIC PROCESSES PROJECT: Regime-Switching OU + Leading Indicators + Order Flow
"""

from typing import List, Dict
import math

from trading_data_structures import (
    COMPANY_SYMBOLS,
    EXTERNAL_SYMBOLS,
    PortfolioState,
    Order,
)


class TradingStrategy:
    def __init__(self):
        self.t = 0

        # --- stochastic state ---
        self.prev_price = {}          # μ_t (previous tick price)
        self.ewma_var = {}            # σ² via EWMA (volatility)
        self.ext_prev = {}
        self.ext_ema = {}             # leading indicator (EWMA returns of externals)

        # parameters
        self.vol_halflife = 120
        self.vol_alpha = 1.0 - math.exp(math.log(0.5) / self.vol_halflife)

        self.ext_halflife = 35
        self.ext_alpha = 1.0 - math.exp(math.log(0.5) / self.ext_halflife)

        self.base_threshold = 0.0054          # همان آستانه‌ای که ۱۶۶۳۸٪ سود داد
        self.stable_multiplier = 1     # در فاز stable حساس‌تر (ترید بیشتر)
        self.volatile_multiplier = 0.005   # در فاز volatile سخت‌تر (جلوگیری از نویز)

    def on_tick(self, assets_map: Dict[str, "AssetData"], portfolio: PortfolioState) -> List[Order]:
        self.t += 1
        orders: List[Order] = []

        # ================== 1. Update Leading Indicators (External) ==================
        ext_momentum = 0.0
        n_ext = 0
        for ext in EXTERNAL_SYMBOLS:
            asset = assets_map.get(ext)
            if not asset or not hasattr(asset, "price") or asset.price <= 0:
                continue

            prev = self.ext_prev.get(ext)
            if prev is None or prev <= 0:
                self.ext_prev[ext] = asset.price
                self.ext_ema[ext] = 0.0
                continue

            r = math.log(asset.price / prev)
            ema = self.ext_ema.get(ext, 0.0)
            ema = (1.0 - self.ext_alpha) * ema + self.ext_alpha * r
            self.ext_ema[ext] = ema
            self.ext_prev[ext] = asset.price

            ext_momentum += ema
            n_ext += 1

        ext_momentum = ext_momentum / max(1, n_ext)   # global leading drift

        # ================== 2. Helpers ==================
        def get_queues(asset):
            buy_q = getattr(asset, "buy_queue", getattr(asset, "buyQueue", []))
            sell_q = getattr(asset, "sell_queue", getattr(asset, "sellQueue", []))
            return buy_q or [], sell_q or []

        def level_price_size(lvl):
            if lvl is None: return None
            if hasattr(lvl, "price") and hasattr(lvl, "size"):
                return float(lvl.price), float(lvl.size)
            if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                return float(lvl[0]), float(lvl[1])
            return None

        def total_liquidity(q):
            return sum(max(0.0, level_price_size(lvl)[1]) for lvl in q if level_price_size(lvl))

        # ================== 3. Main Trading Loop ==================
        for sym in COMPANY_SYMBOLS:
            asset = assets_map.get(sym)
            if not asset: continue

            buy_q, sell_q = get_queues(asset)
            if not buy_q or not sell_q: continue

            bid = level_price_size(buy_q[0])
            ask = level_price_size(sell_q[0])
            if not bid or not ask or ask[0] <= bid[0]: continue

            price = getattr(asset, "price", 0.5 * (bid[0] + ask[0]))
            if price <= 0: continue

            # --- stochastic updates ---
            prev = self.prev_price.get(sym)
            if prev is None or prev <= 0:
                self.prev_price[sym] = price
                self.ewma_var[sym] = 1e-8
                continue

            change = (price - prev) / prev
            r = math.log(price / prev) if price > 0 and prev > 0 else 0.0

            v = self.ewma_var.get(sym, 1e-8)
            v = (1.0 - self.vol_alpha) * v + self.vol_alpha * (r * r)
            self.ewma_var[sym] = max(v, 1e-12)
            vol = math.sqrt(self.ewma_var[sym])

            self.prev_price[sym] = price

            # --- regime detection (Market Phases) ---
            is_stable = vol < 0.005
            threshold = self.base_threshold * (self.stable_multiplier if is_stable else self.volatile_multiplier)

            # --- excess change (Elasticity + Leading Indicators) ---
            excess_change = change - ext_momentum

            pos = int(portfolio.positions.get(sym, 0))

            # ================== ENTRY (Supply/Demand + Order Flow) ==================
            if excess_change < -threshold:                    # قیمت خیلی بیشتر از انتظار (leading) افت کرده → LONG
                ask_liq = total_liquidity(sell_q)
                max_afford = int(portfolio.cash // ask[0]) if ask[0] > 0 else 0
                qty = min(max_afford, int(ask_liq * 0.95))    # تقریباً کل liquidity موجود
                if qty > 0:
                    orders.append(Order(symbol=sym, quantity=qty, price=ask[0]))

            elif excess_change > threshold:                   # قیمت خیلی بیشتر از انتظار بالا رفته → SHORT
                if pos <= 0: continue
                bid_liq = total_liquidity(buy_q)
                qty = min(pos, int(bid_liq * 0.95))
                if qty > 0:
                    orders.append(Order(symbol=sym, quantity=-qty, price=bid[0]))

        return orders