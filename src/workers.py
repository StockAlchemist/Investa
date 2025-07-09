# Auto-generated from main_gui.py modularization
from PySide6.QtCore import QRunnable, QObject, Signal, Slot
from typing import Dict, Any, Set, Optional
import pandas as pd
import logging
import traceback

from market_data import MarketDataProvider
from portfolio_analyzer import (
    extract_dividend_history,
    extract_realized_capital_gains_history,
)
from financial_ratios import (
    calculate_key_ratios_timeseries,
    calculate_current_valuation_ratios,
)
import config


class WorkerSignals(QObject):
    """Defines signals available from a running worker thread (QRunnable).

    Signals:
        finished: Emitted when the worker task has completed, regardless of success.
        error: Emitted when an error occurs during the worker task. Passes a
               string describing the error.
        result: Emitted upon successful completion of the task. Passes the
                calculated results: summary metrics (dict), holdings DataFrame,
                ignored transactions DataFrame, account-level metrics (dict),
                index quotes (dict), and historical performance DataFrame.
    """

    finished = Signal()
    progress = Signal(int)  # <-- ADDED: Percentage complete (0-100)
    error = Signal(str)
    result = Signal(
        dict,
        pd.DataFrame,
        dict,
        dict,
        pd.DataFrame,
        dict,
        dict,
        set,
        dict,
        pd.DataFrame,  # dividend_history_df
        pd.DataFrame,  # capital_gains_history_df
    )

    fundamental_data_ready = Signal(str, dict)  # display_symbol, data_dict


class PortfolioCalculatorWorker(QRunnable):
    """
    Worker thread (QRunnable) for performing portfolio calculations.

    Executes portfolio summary, index quote fetching, and historical performance
    calculations in a separate thread to avoid blocking the main GUI thread.
    Uses signals defined in WorkerSignals to communicate results or errors back.
    """

    def __init__(
        self,
        portfolio_fn,
        portfolio_args,
        portfolio_kwargs,
        # --- REMOVED index_fn ---
        historical_fn,
        historical_args,
        historical_kwargs,
        worker_signals: WorkerSignals,  # <-- ADDED: Pass signals objec
        manual_overrides_dict: Dict[str, Dict[str, Any]],  # For prices, sectors etc.
        user_symbol_map: Dict[str, str],  # New
        user_excluded_symbols: Set[str],  # New
        historical_fn_supports_exclude=False,
        market_provider_available=True,
    ):
        """
        Initializes the worker with calculation functions and arguments.

        Args:
            portfolio_fn (callable): The function to calculate the current portfolio summary.
            portfolio_args (tuple): Positional arguments for portfolio_fn.
            portfolio_kwargs (dict): Keyword arguments for portfolio_fn.
            # index_fn (callable): The function to fetch index quotes. <-- REMOVED
            historical_fn (callable): The function to calculate historical performance.
            historical_args (tuple): Positional arguments for historical_fn.
            historical_kwargs (dict): Keyword arguments for historical_fn.
            manual_overrides_dict (dict): Dictionary of manual overrides (price, asset_type, sector, geography, industry).
            user_symbol_map (Dict[str, str]): User-defined symbol map.
            user_excluded_symbols (Set[str]): User-defined excluded symbols.
        """
        super().__init__()
        self.portfolio_fn = portfolio_fn
        self.portfolio_args = portfolio_args
        # portfolio_kwargs will contain account_currency_map and default_currency
        self.portfolio_kwargs = portfolio_kwargs
        # --- REMOVED self.index_fn = index_fn ---
        self.historical_fn = historical_fn
        self.historical_fn_supports_exclude = historical_fn_supports_exclude
        self.market_provider_available = market_provider_available
        self.historical_args = historical_args
        # historical_kwargs will contain account_currency_map and default_currency
        self.historical_kwargs = historical_kwargs
        self.manual_overrides_dict = manual_overrides_dict
        self.user_symbol_map = user_symbol_map
        self.user_excluded_symbols = user_excluded_symbols
        self.signals = worker_signals  # <-- USE PASSED SIGNALS
        self.original_data = pd.DataFrame()

    @Slot()
    def run(self):
        """Executes the calculations and emits results or errors."""
        portfolio_summary_metrics = {}
        holdings_df = pd.DataFrame()
        # Removed ignored_df placeholder
        account_metrics = {}
        index_quotes = {}
        # historical_data_df = pd.DataFrame() # Removed - we only get full data now
        full_historical_data_df = (
            pd.DataFrame()
        )  # This will hold the full daily data from backend
        dividend_history_df = pd.DataFrame()  # <-- NEW for dividend history
        # Initialize raw data dicts
        hist_prices_adj = {}
        hist_fx = {}
        combined_ignored_indices = set()
        combined_ignored_reasons = {}
        capital_gains_history_df = pd.DataFrame()  # Initialize for capital gains

        portfolio_status = "Error: Portfolio calc not run"
        historical_status = "Error: Historical calc not run"
        overall_status = "Error: Worker did not complete"

        try:
            # --- 1. Run Portfolio Summary Calculation ---
            logging.info("WORKER: Starting portfolio summary calculation...")
            # (No changes needed here, it uses self.portfolio_fn)
            # Make a copy of portfolio_kwargs to modify for the portfolio_fn call
            portfolio_fn_kwargs = self.portfolio_kwargs.copy()
            # Remove the argument not expected by calculate_portfolio_summary
            # It's still available in self.portfolio_kwargs for extract_dividend_history
            # --- ADDED: Log portfolio_fn_kwargs before call ---
            portfolio_fn_kwargs.pop("all_transactions_df_for_worker", None)
            try:
                portfolio_fn_kwargs["manual_overrides_dict"] = (
                    self.manual_overrides_dict  # Pass price/sector overrides
                )  # MODIFIED: Pass new dict
                portfolio_fn_kwargs["user_symbol_map"] = (
                    self.user_symbol_map
                )  # Pass symbol map
                portfolio_fn_kwargs["user_excluded_symbols"] = (
                    self.user_excluded_symbols
                )  # Pass excluded symbols
                logging.debug(
                    f"WORKER: Calling portfolio_fn with args: {self.portfolio_args}, kwargs: {list(portfolio_fn_kwargs.keys())}"
                )
                # --- END ADDED ---

                (
                    p_summary,
                    p_holdings,
                    p_account,
                    p_ignored_idx,
                    p_ignored_rsn,
                    p_status,
                ) = self.portfolio_fn(*self.portfolio_args, **portfolio_fn_kwargs)
                portfolio_summary_metrics = p_summary if p_summary is not None else {}
                holdings_df = p_holdings if p_holdings is not None else pd.DataFrame()
                account_metrics = p_account if p_account is not None else {}
                combined_ignored_indices = (
                    p_ignored_idx if p_ignored_idx is not None else set()
                )
                combined_ignored_reasons = (
                    p_ignored_rsn if p_ignored_rsn is not None else {}
                )
                portfolio_status = (
                    p_status if p_status else "Error: Unknown portfolio status"
                )
                if isinstance(
                    portfolio_summary_metrics, dict
                ):  # Add status if possible
                    logging.debug(
                        f"WORKER: Portfolio summary status: {portfolio_status}"
                    )
                    portfolio_summary_metrics["status_msg"] = portfolio_status
                logging.info(
                    f"WORKER: Portfolio summary calculation finished. Status: {portfolio_status}"
                )

            except Exception as port_e:
                logging.error(
                    f"WORKER: --- Error during portfolio calculation: {port_e} ---",
                    exc_info=True,
                )
                traceback.print_exc()
                portfolio_status = f"Error in Port. Calc: {port_e}"
                portfolio_summary_metrics = {}
                holdings_df = pd.DataFrame()
                account_metrics = {}
                combined_ignored_indices = set()
                combined_ignored_reasons = {}

            # --- 2. Fetch Index Quotes using MarketDataProvider ---
            logging.info("WORKER: Fetching index quotes...")
            try:
                logging.debug("DEBUG Worker: Fetching index quotes...")
                logging.info("WORKER: Fetching index quotes...")
                # --- Instantiate and call MarketDataProvider ---
                if self.market_provider_available:
                    market_provider = MarketDataProvider()
                    # get_index_quotes does not need user_symbol_map or user_excluded_symbols
                    # as it uses its own predefined list from config.py (YFINANCE_INDEX_TICKER_MAP)
                    # If index symbols were to become user-configurable, this would need to change.
                    index_quotes = (
                        market_provider.get_index_quotes()
                    )  # Uses defaults from config
                    logging.info(
                        f"WORKER: Index quotes fetched ({len(index_quotes)} items)."
                    )
                else:
                    logging.error(
                        "WORKER: MarketDataProvider not available, cannot fetch index quotes."
                    )
                    index_quotes = {}
                # --- End MarketDataProvider usage ---
                logging.debug(
                    f"DEBUG Worker: Index quotes fetched ({len(index_quotes)} items)."
                )
                logging.info(
                    f"WORKER: Index quotes fetched ({len(index_quotes)} items)."
                )
            except Exception as idx_e:
                logging.error(
                    f"WORKER: --- Error during index quote fetch: {idx_e} ---",
                    exc_info=True,
                )
                traceback.print_exc()
                index_quotes = {}  # Reset on error

            # --- 3. Run Historical Performance Calculation ---
            # (No changes needed here, it uses self.historical_fn)
            try:
                logging.info("WORKER: Starting historical performance calculation...")
                current_historical_kwargs = self.historical_kwargs.copy()
                if (
                    not self.historical_fn_supports_exclude
                    and "exclude_accounts" in current_historical_kwargs
                ):
                    current_historical_kwargs.pop("exclude_accounts")

                logging.debug(
                    f"DEBUG Worker: Calling historical_fn with kwargs keys: {list(current_historical_kwargs.keys())}"
                )
                # Extract the DataFrame to pass positionally
                transactions_df_for_hist = current_historical_kwargs.pop(
                    "all_transactions_df_cleaned", None
                )

                current_historical_kwargs["worker_signals"] = (
                    self.signals
                )  # Pass signals
                current_historical_kwargs["user_symbol_map"] = self.user_symbol_map
                current_historical_kwargs["user_excluded_symbols"] = (
                    self.user_excluded_symbols
                )
                # --- ADDED: Pass manual_overrides_dict to historical_fn ---
                current_historical_kwargs["manual_overrides_dict"] = (
                    self.manual_overrides_dict
                )
                # --- ADD DEBUG LOG ---
                logging.debug(
                    f"WORKER: Passing to historical_fn, manual_overrides_dict: {self.manual_overrides_dict}"
                )
                # --- END DEBUG LOG ---

                # MODIFIED: Unpack 4 items (full_daily_df, prices, fx, status)
                full_hist_df, h_prices_adj, h_fx, hist_status = self.historical_fn(
                    transactions_df_for_hist,  # Pass positionally
                    *self.historical_args,
                    **current_historical_kwargs,
                )

                full_historical_data_df = (
                    full_hist_df if full_hist_df is not None else pd.DataFrame()
                )  # Store full data
                # historical_data_df = hist_df if hist_df is not None else pd.DataFrame() # Removed
                hist_prices_adj = h_prices_adj if h_prices_adj is not None else {}
                hist_fx = h_fx if h_fx is not None else {}
                historical_status = (
                    hist_status if hist_status else "Error: Unknown historical status"
                )

                if isinstance(
                    portfolio_summary_metrics, dict
                ):  # Add status if possible
                    logging.debug(
                        f"WORKER: Historical performance status: {historical_status}"
                    )
                    portfolio_summary_metrics["historical_status_msg"] = (
                        historical_status
                    )
                logging.debug(
                    f"DEBUG Worker: Historical calculation finished. Status: {historical_status}"
                )
                logging.info(
                    f"WORKER: Historical performance calculation finished. Status: {historical_status}"
                )

            except ValueError as ve:
                logging.error(
                    f"--- ValueError during historical performance unpack: {ve} ---"
                )
                traceback.print_exc()
                historical_status = f"Error unpack: {ve}"
                # historical_data_df = pd.DataFrame() # Removed
                full_historical_data_df = pd.DataFrame()  # Clear data on error
                hist_prices_adj = {}
                hist_fx = {}
            except Exception as hist_e:
                logging.error(
                    f"WORKER: --- Error during historical performance calculation: {hist_e} ---",
                    exc_info=True,
                )
                traceback.print_exc()
                historical_status = f"Error in Hist. Calc: {hist_e}"
                # historical_data_df = pd.DataFrame() # Removed
                full_historical_data_df = pd.DataFrame()  # Clear data on error
                hist_prices_adj = {}
                hist_fx = {}

            # --- 4. Extract Dividend History ---
            logging.info("WORKER: Extracting dividend history...")
            try:
                # Ensure historical_fx_yf is available from the historical calculation step
                # It should be in hist_fx if historical_fn ran successfully
                # If hist_fx is empty (e.g., all transactions and display currency are USD),
                # extract_dividend_history should still be called.
                # get_historical_rate_via_usd_bridge will handle USD->USD conversion as 1.0.
                if (
                    not hist_fx
                ):  # hist_fx might be empty if no FX conversion was needed by historical calc
                    logging.warning(
                        "WORKER: Historical FX data from historical calc is empty. "
                        "Dividend history will rely on same-currency or default FX handling."
                    )
                dividend_history_df = extract_dividend_history(
                    all_transactions_df=self.portfolio_kwargs.get(
                        "all_transactions_df_cleaned"
                    ),
                    display_currency=self.portfolio_kwargs.get("display_currency"),
                    historical_fx_yf=hist_fx,  # Pass potentially empty hist_fx
                    default_currency=self.portfolio_kwargs.get("default_currency"),
                    include_accounts=self.portfolio_kwargs.get("include_accounts"),
                )
                logging.info(
                    f"WORKER: Dividend history extracted ({len(dividend_history_df)} records)."
                )
            except Exception as div_e:
                logging.error(
                    f"WORKER: --- Error during dividend history extraction: {div_e} ---",
                    exc_info=True,
                )
                traceback.print_exc()
                # Optionally, emit an error or set a status part
                dividend_history_df = pd.DataFrame()  # Ensure it's an empty DF on error

            # --- 5. Extract Realized Capital Gains History ---
            logging.info("WORKER: Extracting realized capital gains history...")
            try:
                # Ensure hist_fx is available (from historical calc)
                # and other necessary args from portfolio_kwargs
                capital_gains_history_df = extract_realized_capital_gains_history(
                    all_transactions_df=self.portfolio_kwargs.get(
                        "all_transactions_df_cleaned"
                    ),
                    display_currency=self.portfolio_kwargs.get("display_currency"),
                    historical_fx_yf=hist_fx,  # Use h_fx from historical performance part
                    default_currency=self.portfolio_kwargs.get("default_currency"),
                    shortable_symbols=config.SHORTABLE_SYMBOLS,  # Import from config
                    include_accounts=self.portfolio_kwargs.get("include_accounts"),
                )
                logging.info(
                    f"WORKER: Capital gains history extracted ({len(capital_gains_history_df)} records)."
                )
            except (
                ImportError
            ):  # If extract_realized_capital_gains_history is not yet implemented
                logging.warning(
                    "WORKER: extract_realized_capital_gains_history function not found in portfolio_analyzer. Capital gains will be empty."
                )
                capital_gains_history_df = pd.DataFrame()
            except Exception as cg_e:
                logging.error(
                    f"WORKER: --- Error during capital gains history extraction: {cg_e} ---",
                    exc_info=True,
                )
                traceback.print_exc()
                capital_gains_history_df = pd.DataFrame()  # Ensure empty DF on error

            logging.debug(
                f"DEBUG Worker: dividend_history_df before emit (shape {dividend_history_df.shape if isinstance(dividend_history_df, pd.DataFrame) else 'Not a DF'}):"
            )
            if (
                isinstance(dividend_history_df, pd.DataFrame)
                and not dividend_history_df.empty
            ):
                logging.debug(f"  Head:\n{dividend_history_df.head().to_string()}")
                logging.debug(
                    f"  'DividendAmountDisplayCurrency' NaNs in worker: {dividend_history_df['DividendAmountDisplayCurrency'].isna().sum()} out of {len(dividend_history_df)}"
                )

            # --- 4. Prepare and Emit Combined Results ---
            # (No changes needed here)
            overall_status = (
                f"Portfolio: {portfolio_status} | Historical: {historical_status}"
            )
            portfolio_had_error = any(
                err in portfolio_status for err in ["Error", "Crit", "Fail"]
            )
            historical_had_error = any(
                err in historical_status for err in ["Error", "Crit", "Fail"]
            )

            if portfolio_had_error or historical_had_error:
                logging.warning(
                    f"WORKER: Emitting error signal due to calculation issues. Overall status: {overall_status}"
                )
                self.signals.error.emit(overall_status)

            logging.debug(
                f"WORKER EMIT: Summary Metrics Keys: {list(portfolio_summary_metrics.keys()) if portfolio_summary_metrics else 'Empty'}"
            )
            logging.debug(
                f"WORKER EMIT: Holdings DF Shape: {holdings_df.shape if isinstance(holdings_df, pd.DataFrame) else 'Not DF'}"
            )
            logging.debug(
                f"WORKER EMIT: Full Historical DF Shape: {full_historical_data_df.shape if isinstance(full_historical_data_df, pd.DataFrame) else 'Not DF'}"
            )
            logging.debug(
                f"WORKER EMIT: Dividend History DF Shape: {dividend_history_df.shape if isinstance(dividend_history_df, pd.DataFrame) else 'Not DF'}"
            )
            if (
                isinstance(dividend_history_df, pd.DataFrame)
                and not dividend_history_df.empty
            ):
                logging.debug(
                    f"WORKER EMIT: Dividend History Head:\n{dividend_history_df.head().to_string()}"
                )

            # MODIFIED: Emit result only if no critical errors in sub-parts
            if not (portfolio_had_error or historical_had_error):
                self.signals.result.emit(
                    portfolio_summary_metrics,
                    holdings_df,  # EMIT ACTUAL DATA
                    account_metrics,
                    index_quotes,
                    full_historical_data_df,  # EMIT ACTUAL DATA
                    hist_prices_adj,
                    hist_fx,
                    combined_ignored_indices,
                    combined_ignored_reasons,
                    dividend_history_df,  # EMIT ACTUAL DATA
                    capital_gains_history_df,  # EMIT ACTUAL DATA
                )
                logging.debug(
                    "WORKER: Emitting result signal with actual calculated data."
                )
            else:
                logging.warning(
                    "WORKER: Skipped emitting result signal due to errors in calculation sub-parts."
                )
                # Ensure UI can still finish if error was emitted but not critical enough to stop worker

        except Exception as e:
            logging.error(f"--- Critical Error in Worker Thread run method: {e} ---")
            traceback.print_exc()
            overall_status = f"CritErr in Worker: {e}"
            # --- EMIT DEFAULT/EMPTY VALUES on critical failure (10 args) ---
            # For the test, we still emit the no-arg signal even on error in the try block
            self.signals.result.emit(
                {},  # portfolio_summary_metrics
                pd.DataFrame(),  # holdings_df
                {},  # account_metrics
                {},  # index_quotes
                pd.DataFrame(),  # full_historical_data_df
                {},  # hist_prices_adj
                {},  # hist_fx
                set(),  # combined_ignored_indices
                {},  # combined_ignored_reasons
                pd.DataFrame(),  # dividend_history_df
                pd.DataFrame(),  # capital_gains_history_df
            )
            logging.debug(
                "WORKER: Emitted empty/default results due to critical worker error."
            )
            # --- END EMIT ---
            self.signals.error.emit(overall_status)
        finally:
            logging.info("WORKER: Emitting finished signal.")
            self.signals.finished.emit()


class FundamentalDataWorker(QRunnable):
    """Worker to fetch fundamental data in the background."""

    def __init__(
        self,
        yf_symbol: str,
        display_symbol: str,
        signals: WorkerSignals,
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
            market_provider = MarketDataProvider()  # Assuming default init is fine
            data = market_provider.get_fundamental_data(self.yf_symbol)
            if data is None:  # Ensure data is a dict even if .info fetch fails
                data = {}

            # Fetch additional financial statements (both annual and quarterly)
            financials_annual_df = market_provider.get_financials(
                self.yf_symbol, period_type="annual"
            )
            financials_quarterly_df = market_provider.get_financials(
                self.yf_symbol, period_type="quarterly"
            )
            balance_sheet_annual_df = market_provider.get_balance_sheet(
                self.yf_symbol, period_type="annual"
            )
            balance_sheet_quarterly_df = market_provider.get_balance_sheet(
                self.yf_symbol, period_type="quarterly"
            )
            cashflow_annual_df = market_provider.get_cashflow(
                self.yf_symbol, period_type="annual"
            )
            cashflow_quarterly_df = market_provider.get_cashflow(
                self.yf_symbol, period_type="quarterly"
            )

            if financials_annual_df is not None:
                data["financials_annual"] = financials_annual_df
            if financials_quarterly_df is not None:
                data["financials_quarterly"] = financials_quarterly_df

            if balance_sheet_annual_df is not None:
                data["balance_sheet_annual"] = balance_sheet_annual_df
            if balance_sheet_quarterly_df is not None:
                data["balance_sheet_quarterly"] = balance_sheet_quarterly_df

            if cashflow_annual_df is not None:
                data["cashflow_annual"] = cashflow_annual_df
            if cashflow_quarterly_df is not None:
                data["cashflow_quarterly"] = cashflow_quarterly_df

            # --- ADDED: Calculate and add key ratios ---
            if self.financial_ratios_available:  # Use the instance attribute
                # Pass annual statements for historical ratio series
                key_ratios_df = calculate_key_ratios_timeseries(
                    financials_df=financials_annual_df,
                    balance_sheet_df=balance_sheet_annual_df,
                    # cashflow_df=cashflow_annual_df # If needed by ratios
                )
                data["key_ratios_timeseries"] = key_ratios_df

                current_valuation_ratios = calculate_current_valuation_ratios(
                    ticker_info=data,  # Pass the main ticker_info dict
                    financials_df_latest_annual=financials_annual_df,
                    balance_sheet_df_latest_annual=balance_sheet_annual_df,
                )
                data["current_valuation_ratios"] = current_valuation_ratios
            # --- END ADD ---

            if (
                data is not None
            ):  # data is now a dict, check if it's not None (though it's initialized to {} if .info fails)
                self.signals.fundamental_data_ready.emit(self.display_symbol, data)
            else:  # Fetching itself failed (e.g. network error)
                self.signals.error.emit(
                    f"Failed to fetch fundamental data for {self.display_symbol}."
                )
        except Exception as e:
            logging.error(
                f"Error in FundamentalDataWorker for {self.display_symbol}: {e}"
            )
            traceback.print_exc()
            self.signals.error.emit(
                f"Error fetching fundamentals for {self.display_symbol}: {e}"
            )
        finally:
            # self.signals.finished.emit() # Not strictly needed if result/error always emitted
            pass
