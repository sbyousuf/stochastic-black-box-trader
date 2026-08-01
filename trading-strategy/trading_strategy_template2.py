"""
================================================================================
STOCHASTIC PROCESSES PROJECT: ALGORITHMIC TRADING STRATEGY
================================================================================

Student Name: [YOUR NAME]
Student ID:   [YOUR ID]

INSTRUCTIONS:
1. Implement your logic in the `on_tick()` method.
2. Only symbols in `COMPANY_SYMBOLS` are tradeable.
3. Market liquidity is finite. Check the `size` of entries in the queues.
================================================================================
"""
from typing import List, Dict

from trading_data_structures import (
    COMPANY_SYMBOLS,
    EXTERNAL_SYMBOLS,
    PortfolioState,
    Order,
)

class TradingStrategy:
    def __init__(self):
        """Called once at the start of the simulation."""
        # store the last price for each symbol – enough to detect a price move
        self.prev_price: Dict[str, float] = {}

    def on_tick(self, assets_map: Dict[str, "AssetData"], portfolio: PortfolioState) -> List[Order]:
        """
        Processes a single market tick.
        Return a list of orders for the current tick.
        """
        orders: List[Order] = []

        # Simple momentum‑based example: buy on price drops, sell on price rises
        for sym in COMPANY_SYMBOLS:
            asset = assets_map[sym]
            if asset is None:
                continue

            if sym not in self.prev_price:
                change = 0.0
            else:
                change = (asset.price - self.prev_price[sym]) / self.prev_price[sym]

            if change < 0.0054:
                max_afford = int(portfolio.cash // asset.price)
                max_liquid = int(sum(entry.size for entry in asset.sell_queue))

                qty = min(max_afford, max_liquid)          # buy as much as possible
                if qty > 0:
                    best_ask_price = asset.sell_queue[0].price
                    orders.append(Order(symbol=sym, quantity=qty, price=best_ask_price))

                    portfolio.cash -= qty * best_ask_price
                    portfolio.positions[sym] = portfolio.positions.get(sym, 0) + qty

            elif change > 0.0054:
                held = portfolio.positions.get(sym, 0)
                if held <= 0:
                    continue

                max_liquid = int(sum(entry.size for entry in asset.buy_queue))

                qty = min(held, max_liquid)                # sell as much as possible
                if qty > 0:
                    best_bid_price = asset.buy_queue[0].price
                    orders.append(Order(symbol=sym, quantity=-qty, price=best_bid_price))

                    portfolio.cash += qty * best_bid_price
                    portfolio.positions[sym] = held - qty

            self.prev_price[sym] = asset.price

        return orders


#Initial Balance: 99875.96
#Final Balance:   4953793375.81
#Profit:          4953693499.85 (4959845.69%)
#Total Trades:    239331
#Total Ticks:     43800