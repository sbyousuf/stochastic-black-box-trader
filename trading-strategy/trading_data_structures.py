"""
================================================================================
STOCHASTIC PROCESSES PROJECT: ALGORITHMIC TRADING STRATEGY - DATA STRUCTURES
================================================================================

This file contains all data structures and constants for the trading strategy.
Do NOT modify class names or method signatures.

NOTE: COMPANY_SYMBOLS and EXTERNAL_SYMBOLS are dynamically configured
at runtime. These defaults are for local testing only.
================================================================================
"""

from dataclasses import dataclass
from typing import Dict, Optional, List

# =============================================================================
# MARKET CONFIGURATION (DYNAMICALLY CONFIGURED AT RUNTIME)
# =============================================================================

COMPANY_SYMBOLS: List[str] = ["COMP_0", "COMP_1", "COMP_2"]
EXTERNAL_SYMBOLS: List[str] = ["EXT_0"]

# =============================================================================
# DATA STRUCTURES (DO NOT MODIFY)
# =============================================================================

@dataclass
class OrderBookEntry:
    """A single level in the market depth."""
    price: float
    size: float

@dataclass
class AssetData:
    """Market data for a single asset at a point in time."""
    price: float                   # Reference/Last mid-price
    buy_queue: List[OrderBookEntry] # Pending buy orders (sorted descending)
    sell_queue: List[OrderBookEntry] # Pending sell orders (sorted ascending)


@dataclass
class PortfolioState:
    """Current state of your trading account."""
    cash: float
    positions: Dict[str, int]

@dataclass
class Order:
    """A limit order to be submitted to the exchange."""
    symbol: str
    quantity: int    # Positive for BUY, Negative for SELL
    price: float     # Limit Price
