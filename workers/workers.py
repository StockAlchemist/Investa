# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          workers.py
 Purpose:       Background worker classes for Investa Portfolio Dashboard.
 Author:        Kit Matan and Google Gemini
 Created:       30/05/2024
 Copyright:     (c) Kit Matan 2024
 Licence:       MIT
-------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""
import logging
import traceback
import pandas as pd
import numpy as np
from typing import Dict, Any, Set, Optional # Added Optional

from PySide6.QtCore import QObject, Signal, QRunnable, Slot

# --- Configuration (import if needed, or pass values directly) ---
# Assuming these might be needed by workers eventually, or are part of their function signatures
try:
    import config # For SHORTABLE_SYMBOLS etc.
    from portfolio_logic import (
        calculate_portfolio_summary,
        CASH_SYMBOL_CSV,
        calculate_historical_performance,
        extract_dividend_history,
        extract_realized_capital_gains_history
    )
    from market_data import MarketDataProvider # For both workers
    from financial_ratios import ( # For FundamentalDataWorker
        calculate_key_ratios_timeseries,
        calculate_current_valuation_ratios
    )
    # These flags would ideally be passed or determined, not imported if they can change
    MARKET_PROVIDER_AVAILABLE = True # Placeholder, should be determined by main app
    HISTORICAL_FN_SUPPORTS_EXCLUDE = True # Placeholder
    FINANCIAL_RATIOS_AVAILABLE = True # Placeholder
except ImportError as e:
    logging.error(f"workers.py: Error importing domain logic modules: {e}")
    # Define fallbacks if necessary for the classes to be syntactically valid
    CASH_SYMBOL_CSV = "__CASH__"
    MARKET_PROVIDER_AVAILABLE = False
    HISTORICAL_FN_SUPPORTS_EXCLUDE = False
    FINANCIAL_RATIOS_AVAILABLE = False
    # Dummy functions if needed for type hinting or basic structure
    def calculate_portfolio_summary(*args, **kwargs): return {}, pd.DataFrame(), {}, {}, {}, "Error"
    def calculate_historical_performance(*args, **kwargs): return pd.DataFrame(), {}, {}, "Error"
    def extract_dividend_history(*args, **kwargs): return pd.DataFrame()
    def extract_realized_capital_gains_history(*args, **kwargs): return pd.DataFrame()
    class MarketDataProvider: pass
    def calculate_key_ratios_timeseries(*args, **kwargs): return pd.DataFrame()
    def calculate_current_valuation_ratios(*args, **kwargs): return {}


class PortfolioWorkerSignals(QObject):
    """
    Defines signals available from the PortfolioCalculatorWorker.
    """
    finished = Signal()
    progress = Signal(int)
    error = Signal(str)
    # Signature for portfolio calculation results:
    result = Signal(
        dict,  # portfolio_summary_metrics
        pd.DataFrame,  # holdings_df
        dict,  # account_metrics
        dict,  # index_quotes
        pd.DataFrame,  # full_historical_data_df
        dict,  # hist_prices_adj
        dict,  # hist_fx
        set,  # combined_ignored_indices
        dict,  # combined_ignored_reasons
        pd.DataFrame,  # dividend_history_df
        pd.DataFrame,  # capital_gains_history_df
    )

class FundamentalDataWorkerSignals(QObject):
    """
    Defines signals available from the FundamentalDataWorker.
    """
    finished = Signal() # Though not strictly used by FundamentalDataWorker's current logic to emit
    error = Signal(str)
    fundamental_data_ready = Signal(str, dict) # display_symbol, data_dict


class PortfolioCalculatorWorker(QRunnable):
    """
    Worker thread (QRunnable) for performing portfolio calculations.
    """
    def __init__(
        self,
        portfolio_fn, # calculate_portfolio_summary
        portfolio_args,
        portfolio_kwargs,
        historical_fn, # calculate_historical_performance
        historical_args,
        historical_kwargs,
        worker_signals: PortfolioWorkerSignals, # Specific signals type
        manual_overrides_dict: Dict[str, Dict[str, Any]],
        user_symbol_map: Dict[str, str],
        user_excluded_symbols: Set[str],
    ):
        super().__init__()
        self.portfolio_fn = portfolio_fn
        self.portfolio_args = portfolio_args
        self.portfolio_kwargs = portfolio_kwargs
        self.historical_fn = historical_fn
        self.historical_args = historical_args
        self.historical_kwargs = historical_kwargs
        self.signals = worker_signals
        self.manual_overrides_dict = manual_overrides_dict
        self.user_symbol_map = user_symbol_map
        self.user_excluded_symbols = user_excluded_symbols
        # self.original_data = pd.DataFrame() # Not used within worker

    @Slot()
    def run(self):
        portfolio_summary_metrics = {}
        holdings_df = pd.DataFrame()
        account_metrics = {}
        index_quotes = {}
        full_historical_data_df = pd.DataFrame()
        hist_prices_adj = {}
        hist_fx = {}
        combined_ignored_indices = set()
        combined_ignored_reasons = {}
        dividend_history_df = pd.DataFrame()
        capital_gains_history_df = pd.DataFrame()
        portfolio_status = "Error: Portfolio calc not run"
        historical_status = "Error: Historical calc not run"

        try:
            logging.info("WORKER: Starting portfolio summary calculation...")
            portfolio_fn_kwargs = self.portfolio_kwargs.copy()
            portfolio_fn_kwargs.pop("all_transactions_df_for_worker", None) # No longer used
            portfolio_fn_kwargs["manual_overrides_dict"] = self.manual_overrides_dict
            portfolio_fn_kwargs["user_symbol_map"] = self.user_symbol_map
            portfolio_fn_kwargs["user_excluded_symbols"] = self.user_excluded_symbols

            (p_summary, p_holdings, p_account, p_ignored_idx, p_ignored_rsn, p_status) = \
                self.portfolio_fn(*self.portfolio_args, **portfolio_fn_kwargs)

            portfolio_summary_metrics = p_summary if p_summary is not None else {}
            holdings_df = p_holdings if p_holdings is not None else pd.DataFrame()
            account_metrics = p_account if p_account is not None else {}
            combined_ignored_indices = p_ignored_idx if p_ignored_idx is not None else set()
            combined_ignored_reasons = p_ignored_rsn if p_ignored_rsn is not None else {}
            portfolio_status = p_status if p_status else "Error: Unknown portfolio status"
            if isinstance(portfolio_summary_metrics, dict):
                portfolio_summary_metrics["status_msg"] = portfolio_status
            logging.info(f"WORKER: Portfolio summary calculation finished. Status: {portfolio_status}")

            logging.info("WORKER: Fetching index quotes...")
            if MARKET_PROVIDER_AVAILABLE:
                market_provider = MarketDataProvider()
                index_quotes = market_provider.get_index_quotes()
                logging.info(f"WORKER: Index quotes fetched ({len(index_quotes)} items).")
            else:
                logging.error("WORKER: MarketDataProvider not available.")
                index_quotes = {}

            logging.info("WORKER: Starting historical performance calculation...")
            current_historical_kwargs = self.historical_kwargs.copy()
            if not HISTORICAL_FN_SUPPORTS_EXCLUDE and "exclude_accounts" in current_historical_kwargs:
                current_historical_kwargs.pop("exclude_accounts")

            transactions_df_for_hist = current_historical_kwargs.pop("all_transactions_df_cleaned", None)
            current_historical_kwargs["worker_signals"] = self.signals # Pass signals object
            current_historical_kwargs["user_symbol_map"] = self.user_symbol_map
            current_historical_kwargs["user_excluded_symbols"] = self.user_excluded_symbols
            current_historical_kwargs["manual_overrides_dict"] = self.manual_overrides_dict

            full_hist_df, h_prices_adj, h_fx, hist_status = self.historical_fn(
                transactions_df_for_hist, *self.historical_args, **current_historical_kwargs
            )
            full_historical_data_df = full_hist_df if full_hist_df is not None else pd.DataFrame()
            hist_prices_adj = h_prices_adj if h_prices_adj is not None else {}
            hist_fx = h_fx if h_fx is not None else {}
            historical_status = hist_status if hist_status else "Error: Unknown historical status"
            if isinstance(portfolio_summary_metrics, dict):
                portfolio_summary_metrics["historical_status_msg"] = historical_status
            logging.info(f"WORKER: Historical performance calculation finished. Status: {historical_status}")

            logging.info("WORKER: Extracting dividend history...")
            dividend_history_df = extract_dividend_history(
                all_transactions_df=self.portfolio_kwargs.get("all_transactions_df_cleaned"),
                display_currency=self.portfolio_kwargs.get("display_currency"),
                historical_fx_yf=hist_fx,
                default_currency=self.portfolio_kwargs.get("default_currency"),
                include_accounts=self.portfolio_kwargs.get("include_accounts"),
            )
            logging.info(f"WORKER: Dividend history extracted ({len(dividend_history_df)} records).")

            logging.info("WORKER: Extracting realized capital gains history...")
            capital_gains_history_df = extract_realized_capital_gains_history(
                all_transactions_df=self.portfolio_kwargs.get("all_transactions_df_cleaned"),
                display_currency=self.portfolio_kwargs.get("display_currency"),
                historical_fx_yf=hist_fx,
                default_currency=self.portfolio_kwargs.get("default_currency"),
                shortable_symbols=config.SHORTABLE_SYMBOLS,
                include_accounts=self.portfolio_kwargs.get("include_accounts"),
            )
            logging.info(f"WORKER: Capital gains history extracted ({len(capital_gains_history_df)} records).")

            overall_status = f"Portfolio: {portfolio_status} | Historical: {historical_status}"
            portfolio_had_error = any(err in portfolio_status for err in ["Error", "Crit", "Fail"])
            historical_had_error = any(err in historical_status for err in ["Error", "Crit", "Fail"])

            if portfolio_had_error or historical_had_error:
                self.signals.error.emit(overall_status)
            else:
                self.signals.result.emit(
                    portfolio_summary_metrics, holdings_df, account_metrics, index_quotes,
                    full_historical_data_df, hist_prices_adj, hist_fx,
                    combined_ignored_indices, combined_ignored_reasons,
                    dividend_history_df, capital_gains_history_df
                )
        except Exception as e:
            logging.error(f"--- Critical Error in PortfolioCalculatorWorker run method: {e} ---")
            traceback.print_exc()
            self.signals.error.emit(f"CritErr in Worker: {e}")
            # Emit empty/default results on critical failure
            self.signals.result.emit({}, pd.DataFrame(), {}, {}, pd.DataFrame(), {}, {}, set(), {}, pd.DataFrame(), pd.DataFrame())
        finally:
            self.signals.finished.emit()


class FundamentalDataWorker(QRunnable):
    """Worker to fetch fundamental data in the background."""
    def __init__(
        self,
        yf_symbol: str,
        display_symbol: str,
        signals: FundamentalDataWorkerSignals, # Specific signals type
        financial_ratios_available_flag: bool,
    ):
        super().__init__()
        self.yf_symbol = yf_symbol
        self.display_symbol = display_symbol
        self.signals = signals
        self.financial_ratios_available = financial_ratios_available_flag

    @Slot()
    def run(self):
        try:
            market_provider = MarketDataProvider()
            data = market_provider.get_fundamental_data(self.yf_symbol)
            if data is None: data = {}

            financials_annual_df = market_provider.get_financials(self.yf_symbol, period_type="annual")
            financials_quarterly_df = market_provider.get_financials(self.yf_symbol, period_type="quarterly")
            balance_sheet_annual_df = market_provider.get_balance_sheet(self.yf_symbol, period_type="annual")
            balance_sheet_quarterly_df = market_provider.get_balance_sheet(self.yf_symbol, period_type="quarterly")
            cashflow_annual_df = market_provider.get_cashflow(self.yf_symbol, period_type="annual")
            cashflow_quarterly_df = market_provider.get_cashflow(self.yf_symbol, period_type="quarterly")

            if financials_annual_df is not None: data["financials_annual"] = financials_annual_df
            if financials_quarterly_df is not None: data["financials_quarterly"] = financials_quarterly_df
            if balance_sheet_annual_df is not None: data["balance_sheet_annual"] = balance_sheet_annual_df
            if balance_sheet_quarterly_df is not None: data["balance_sheet_quarterly"] = balance_sheet_quarterly_df
            if cashflow_annual_df is not None: data["cashflow_annual"] = cashflow_annual_df
            if cashflow_quarterly_df is not None: data["cashflow_quarterly"] = cashflow_quarterly_df

            if self.financial_ratios_available:
                key_ratios_df = calculate_key_ratios_timeseries(
                    financials_df=financials_annual_df,
                    balance_sheet_df=balance_sheet_annual_df,
                )
                data["key_ratios_timeseries"] = key_ratios_df

                current_valuation_ratios = calculate_current_valuation_ratios(
                    ticker_info=data,
                    financials_df_latest_annual=financials_annual_df,
                    balance_sheet_df_latest_annual=balance_sheet_annual_df,
                )
                data["current_valuation_ratios"] = current_valuation_ratios

            self.signals.fundamental_data_ready.emit(self.display_symbol, data)

        except Exception as e:
            logging.error(f"Error in FundamentalDataWorker for {self.display_symbol}: {e}")
            traceback.print_exc()
            self.signals.error.emit(f"Error fetching fundamentals for {self.display_symbol}: {e}")
        # finally:
            # self.signals.finished.emit() # Not strictly needed if result/error is always emitted
