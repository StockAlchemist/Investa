"""Stable portfolio contracts, with legacy dynamic fields retained as extras."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Freshness(BaseModel):
    calculated_at: float | None = None
    quotes_as_of: float | None = None
    refreshing: bool = False


class PortfolioMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")
    market_value: float | None = None
    day_change_display: float | None = None
    day_change_percent: float | None = None
    cash_balance: float | None = None
    cumulative_twr: float | None = None
    annualized_twr: float | None = None
    portfolio_mwr: float | None = None
    all_selected_closed: bool = False
    closed_accounts: list[str] = Field(default_factory=list)


class HeadlineResponse(BaseModel):
    metrics: PortfolioMetrics | None = None
    freshness: Freshness | None = None


class SummaryResponse(HeadlineResponse):
    account_metrics: dict[str, Any] | None = None
    summary_df: list[dict[str, Any]] = Field(default_factory=list)
    holdings_dict: dict[str, Any] = Field(default_factory=dict)


class HoldingResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    Symbol: str
    Quantity: float | None = None
    Account: str | None = None
    lots: list[dict[str, Any]] = Field(default_factory=list)


class HistoryPoint(BaseModel):
    model_config = ConfigDict(extra="allow")
    date: str
    value: float
    twr: float
    drawdown: float
    is_baseline: bool = False
    abs_gain: float = 0
    abs_roi: float = 0
    cum_flow: float = 0
