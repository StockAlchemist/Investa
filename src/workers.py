# Auto-generated from main_gui.py modularization
from PySide6.QtCore import QRunnable, QObject, Signal, Slot
from typing import Dict, Any, Set, Optional
import pandas as pd
import logging
import traceback
from datetime import datetime, timedelta

from market_data import MarketDataProvider
from portfolio_analyzer import (
    extract_dividend_history,
    extract_realized_capital_gains_history,
    calculate_correlation_matrix,
    run_scenario_analysis,
)
from factor_analyzer import run_factor_regression
from financial_ratios import (
    calculate_key_ratios_timeseries,
    calculate_current_valuation_ratios,
)
import config
from finutils import map_to_yf_symbol


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
        pd.DataFrame,  # correlation_matrix_df
        dict,  # factor_analysis_results
        dict,  # scenario_analysis_result
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
        historical_fn,
        historical_args,
        historical_kwargs,
        worker_signals: WorkerSignals,
        manual_overrides_dict: Dict[str, Dict[str, Any]],
        user_symbol_map: Dict[str, str],
        user_excluded_symbols: Set[str],
        market_data_provider: MarketDataProvider,  # ADDED
        force_historical_refresh: bool = True,
        historical_fn_supports_exclude=False,
        market_provider_available=True,
        factor_model_name: str = "Fama-French 3-Factor",
        scenario_shocks: Optional[Dict[str, float]] = None,  # <-- ADDED
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
            scenario_shocks (Optional[Dict[str, float]]): User-defined scenario shocks.
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
        self.market_data_provider = market_data_provider
        self.factor_model_name = factor_model_name
        self.scenario_shocks = scenario_shocks  # <-- ADDED
        self.force_historical_refresh = force_historical_refresh
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
                    use_raw_data_cache=not self.force_historical_refresh,
                    use_daily_results_cache=not self.force_historical_refresh,
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
                    logging.info(
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

            # --- 6. Calculate Correlation Matrix ---
            correlation_matrix_df = pd.DataFrame()
            logging.info("WORKER: Calculating correlation matrix...")
            try:
                # Get all unique stock symbols from transactions
                all_transactions_df = self.portfolio_kwargs.get(
                    "all_transactions_df_for_worker"
                )
                logging.debug(
                    f"WORKER: all_transactions_df_for_worker received: {type(all_transactions_df)}, empty: {all_transactions_df.empty if isinstance(all_transactions_df, pd.DataFrame) else 'N/A'}"
                )
                if all_transactions_df is None or all_transactions_df.empty:
                    logging.warning(
                        "WORKER: No transactions data available for correlation matrix calculation."
                    )
                    raise ValueError("No transactions data")

                # Filter for stock/ETF transactions and get unique symbols
                logging.debug(
                    f"DEBUG WORKER: Columns in all_transactions_df: {all_transactions_df.columns.tolist()}"
                )  # Direct print for debugging
                logging.debug(
                    f"WORKER: Columns in all_transactions_df: {all_transactions_df.columns.tolist()}"
                )

                if "Type" not in all_transactions_df.columns:
                    logging.error(
                        "WORKER: 'Type' column missing from transactions DataFrame. Cannot calculate correlation matrix for individual stocks."
                    )
                    raise ValueError("'Type' column missing")

                stock_transactions = all_transactions_df[
                    (
                        all_transactions_df["Type"]
                        .str.lower()
                        .isin(["buy", "sell", "dividend", "split"])
                    )
                    & (all_transactions_df["Symbol"] != config.CASH_SYMBOL_CSV)
                ]
                unique_stock_symbols = stock_transactions["Symbol"].unique().tolist()
                logging.debug(
                    f"WORKER: Unique stock symbols from all transactions: {unique_stock_symbols}"
                )

                # Filter to include only currently held stocks
                if not holdings_df.empty and "Symbol" in holdings_df.columns:
                    currently_held_symbols = holdings_df["Symbol"].unique().tolist()
                    unique_stock_symbols = [
                        sym
                        for sym in unique_stock_symbols
                        if sym in currently_held_symbols
                    ]
                    logging.debug(
                        f"WORKER: Unique stock symbols (currently held only): {unique_stock_symbols}"
                    )
                else:
                    logging.warning(
                        "WORKER: Holdings DataFrame is empty or missing 'Symbol' column. Cannot filter for currently held stocks."
                    )

                if not unique_stock_symbols:
                    logging.info(
                        "WORKER: No stock/ETF symbols found in transactions for correlation matrix."
                    )
                    raise ValueError("No stock symbols")

                # Map internal symbols to YF symbols
                yf_symbols_for_corr = []
                internal_to_yf_map_for_corr = {}
                for internal_sym in unique_stock_symbols:
                    yf_sym = map_to_yf_symbol(
                        internal_sym, self.user_symbol_map, self.user_excluded_symbols
                    )
                    if yf_sym:
                        yf_symbols_for_corr.append(yf_sym)
                        internal_to_yf_map_for_corr[yf_sym] = (
                            internal_sym  # Store YF -> Internal mapping
                        )
                    else:
                        logging.info(
                            f"WORKER: Skipping {internal_sym} for correlation: no YF mapping or excluded."
                        )
                logging.debug(
                    f"WORKER: YF symbols for correlation after mapping: {yf_symbols_for_corr}"
                )

                if not yf_symbols_for_corr:
                    logging.info(
                        "WORKER: No valid YF symbols for correlation matrix after mapping/exclusion."
                    )
                    raise ValueError("No valid YF symbols")

                # Determine date range for historical data
                start_date_corr = all_transactions_df["Date"].min()
                end_date_corr = all_transactions_df["Date"].max()
                if not pd.isna(start_date_corr) and not pd.isna(end_date_corr):
                    start_date_corr = start_date_corr.date()
                    end_date_corr = end_date_corr.date()
                else:
                    logging.warning(
                        "WORKER: Invalid date range for historical data for correlation. Using default."
                    )
                    start_date_corr = datetime.now().date() - timedelta(
                        days=365 * 2
                    )  # Last 2 years
                    end_date_corr = datetime.now().date()

                # Fetch historical prices for individual stocks
                historical_prices_for_corr, fetch_failed_corr = (
                    self.market_data_provider.get_historical_data(
                        symbols_yf=yf_symbols_for_corr,
                        start_date=start_date_corr,
                        end_date=end_date_corr,
                        use_cache=True,
                        cache_key=f"CORR_HIST::{start_date_corr}::{end_date_corr}::{hash(frozenset(yf_symbols_for_corr))}",
                    )
                )
                logging.debug(
                    f"WORKER: Historical prices fetched for correlation: {list(historical_prices_for_corr.keys())}. Fetch failed status: {fetch_failed_corr}"
                )

                if fetch_failed_corr or not historical_prices_for_corr:
                    logging.warning(
                        "WORKER: Failed to fetch historical prices for correlation matrix."
                    )
                    raise ValueError("Historical price fetch failed")

                # Calculate daily returns for each stock
                returns_for_corr = pd.DataFrame()
                for yf_sym, price_df in historical_prices_for_corr.items():
                    if not price_df.empty and "price" in price_df.columns:
                        internal_sym = internal_to_yf_map_for_corr.get(
                            yf_sym, yf_sym
                        )  # Use internal symbol as column name
                        returns_for_corr[internal_sym] = price_df["price"].pct_change()
                logging.debug(
                    f"WORKER: Returns DataFrame before dropna (shape: {returns_for_corr.shape}):\n{returns_for_corr.head().to_string()}"
                )

                if returns_for_corr.empty:
                    logging.warning(
                        "WORKER: No valid returns calculated for correlation matrix."
                    )
                    raise ValueError("No valid returns")

                # Drop rows with NaNs (e.g., first row after pct_change)
                returns_for_corr.dropna(inplace=True)
                logging.debug(
                    f"WORKER: Returns DataFrame after dropna (shape: {returns_for_corr.shape}):\n{returns_for_corr.head().to_string()}"
                )

                # Calculate correlation matrix
                if (
                    returns_for_corr.shape[1] > 1
                ):  # Need at least 2 columns for correlation
                    correlation_matrix_df = calculate_correlation_matrix(
                        returns_for_corr
                    )
                    logging.info(
                        f"WORKER: Correlation matrix calculated (shape: {correlation_matrix_df.shape})."
                    )
                else:
                    logging.info(
                        "WORKER: Not enough stock symbols with valid returns to calculate correlation matrix (>1 needed)."
                    )
                    correlation_matrix_df = (
                        pd.DataFrame()
                    )  # Ensure it's empty if not enough data

            except Exception as corr_e:
                logging.error(
                    f"WORKER: --- Error during correlation matrix calculation: {corr_e} ---",
                    exc_info=True,
                )
                traceback.print_exc()
                correlation_matrix_df = pd.DataFrame()

            # --- 7. Run Factor Analysis ---
            factor_analysis_results = {}
            logging.info("WORKER: Running factor analysis...")
            try:
                # For factor analysis, we need portfolio returns.
                portfolio_returns_series = pd.Series()
                if "Portfolio Value" in full_historical_data_df.columns:
                    df_for_analysis = full_historical_data_df.copy()
                    if "Date" in df_for_analysis.columns and not isinstance(
                        df_for_analysis.index, pd.DatetimeIndex
                    ):
                        df_for_analysis["Date"] = pd.to_datetime(
                            df_for_analysis["Date"]
                        )
                        df_for_analysis.set_index("Date", inplace=True)

                    # Convert portfolio value to periodic returns
                    portfolio_returns_series = (
                        df_for_analysis["Portfolio Value"].pct_change().dropna()
                    )
                    portfolio_returns_series.name = "Portfolio_Returns"

                if not portfolio_returns_series.empty:
                    # Use the first selected benchmark as the market factor
                    benchmark_data_for_factor_analysis = None
                    if self.historical_kwargs.get("benchmark_symbols_yf"):
                        first_benchmark_ticker = self.historical_kwargs[
                            "benchmark_symbols_yf"
                        ][0]
                        benchmark_col_name = f"{first_benchmark_ticker} Price"
                        if benchmark_col_name in df_for_analysis.columns:
                            benchmark_data_for_factor_analysis = df_for_analysis[
                                [benchmark_col_name]
                            ]

                    ff3_results = run_factor_regression(
                        portfolio_returns_series,
                        self.factor_model_name,
                        benchmark_data=benchmark_data_for_factor_analysis,
                    )
                    if ff3_results:
                        # Store relevant parts of the summary, e.g., params, pvalues, rsquared
                        factor_analysis_results = {
                            "params": ff3_results.params.to_dict(),
                            "pvalues": ff3_results.pvalues.to_dict(),
                            "rsquared": ff3_results.rsquared,
                            "summary_text": str(
                                ff3_results.summary()
                            ),  # Store full summary for display
                            "model_name": self.factor_model_name,  # NEW: Store the model name
                        }
                        logging.info("WORKER: Factor analysis completed successfully.")
                    else:
                        logging.warning("WORKER: Factor analysis returned no results.")
                else:
                    logging.warning(
                        "WORKER: No portfolio returns data for factor analysis."
                    )
            except Exception as fa_e:
                logging.error(
                    f"WORKER: --- Error during factor analysis: {fa_e} ---",
                    exc_info=True,
                )
                traceback.print_exc()
                factor_analysis_results = {}

            # --- 8. Run Scenario Analysis (Placeholder - requires UI input) ---
            scenario_analysis_result = {}
            logging.info("WORKER: Running scenario analysis (placeholder)...")
            try:
                # This function typically takes user input (scenario shocks) and factor betas.
                # Since this is a worker, we'll just pass dummy data for now.
                # In a real implementation, scenario_shocks would be passed from the main thread.
                dummy_factor_betas = factor_analysis_results.get(
                    "params", {}
                )  # Use betas from factor analysis
                if "const" in dummy_factor_betas:
                    del dummy_factor_betas["const"]  # Remove intercept

                dummy_portfolio_value = portfolio_summary_metrics.get(
                    "market_value", 0.0
                )

                if (
                    dummy_factor_betas
                    and self.scenario_shocks
                    and dummy_portfolio_value != 0.0
                ):
                    scenario_analysis_result = run_scenario_analysis(
                        factor_betas=dummy_factor_betas,
                        scenario_shocks=self.scenario_shocks,
                        portfolio_value=dummy_portfolio_value,
                    )
                    logging.info("WORKER: Scenario analysis completed.")
                else:
                    logging.info(
                        "WORKER: Skipping scenario analysis due to missing betas, scenario shocks, or portfolio value."
                    )
            except Exception as sa_e:
                logging.error(
                    f"WORKER: --- Error during scenario analysis: {sa_e} ---",
                    exc_info=True,
                )
                traceback.print_exc()
                scenario_analysis_result = {}

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

            # --- Prepare and Emit Combined Results ---
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
                    correlation_matrix_df,  # NEW
                    factor_analysis_results,  # NEW
                    scenario_analysis_result,  # NEW
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
                pd.DataFrame(),  # correlation_matrix_df
                {},  # factor_analysis_results
                {},  # scenario_analysis_result
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
