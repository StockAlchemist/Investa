# Frozen reference: _calculate_accumulated_gains_and_resample from
# src/portfolio_history.py at commit 6386bd79, before the Rust port.
# tests/test_rust_history_parity.py holds rust/investa_core/src/history.rs (and
# the shim around it) to it, bitwise. Never edit this to follow a change in Rust.
# ruff: noqa
import logging
from datetime import date
from typing import List, Tuple

import numpy as np
import pandas as pd


def _calculate_accumulated_gains_and_resample(
    daily_df: pd.DataFrame,
    benchmark_symbols_yf: List[str],
    interval: str,
    start_date_filter: date,  # ADDED
    end_date_filter: date,  # ADDED
) -> Tuple[pd.DataFrame, float, str]:  # Return signature unchanged for now
    """
    Calculates accumulated gains for portfolio and benchmarks, and optionally resamples the data.

    Takes the daily results DataFrame (with 'daily_return' calculated) and computes the
    cumulative gain factor (Time-Weighted Return) for the portfolio. It also calculates
    the cumulative gain factor for each benchmark based on their daily price changes
    (using adjusted prices). If the specified `interval` is 'W' (Weekly) or 'ME' (Month End),
    it resamples the data to that frequency, recalculating accumulated gains based on
    the resampled period returns.

    Args:
        daily_df (pd.DataFrame): DataFrame containing daily results, including 'value',
            'daily_return', and benchmark prices. Must be indexed by date.
        benchmark_symbols_yf (List[str]): List of YF benchmark tickers present in daily_df.
        interval (str): The desired output interval ('D', 'W', or 'ME').
        start_date_filter (date): The start date requested by the user for filtering the final output.
        end_date_filter (date): The end date requested by the user for filtering the final output.

    Returns:
        Tuple[pd.DataFrame, float, str]:
            - final_df_output (pd.DataFrame): DataFrame indexed by the chosen interval's dates,
              containing columns like 'Portfolio Value', 'Portfolio Daily Gain' (summed if resampled),
              'Portfolio Accumulated Gain', and benchmark prices/accumulated gains.
            - final_twr_factor (float): The final cumulative TWR factor for the portfolio over the
              entire period (based on daily data before resampling). np.nan if unavailable.
            - status_update (str): String describing the outcome (calculation, resampling, errors).
    """
    if daily_df.empty:
        return pd.DataFrame(), np.nan, " (No daily data for final calcs)"
    if "daily_return" not in daily_df.columns:
        return pd.DataFrame(), np.nan, " (Daily return column missing)"

    status_update = ""
    final_twr_factor = np.nan
    results_df = daily_df.copy()
    # --- ADDED: Log input DataFrame shape and tail ---
    logging.debug(
        f"[_calculate_accumulated_gains_and_resample] Input daily_df shape: {results_df.shape}"
    )
    logging.debug(f"Input daily_df tail:\n{results_df.tail().to_string()}")

    try:
        gain_factors_portfolio = 1 + results_df["daily_return"].fillna(0.0)
        # --- FIX: Prevent TWR "Death" by Zero ---
        # If any daily return is -1.0 (-100%), the factor becomes 0.0.
        # This causes the cumprod to stay 0.0 forever, ruining the graph for all future dates.
        # We replace factors near 0.0 with 1.0 (neutral 0% return) to bridge these data gaps/errors.
        zero_factor_mask = gain_factors_portfolio < 1e-6
        if zero_factor_mask.any():
            gain_factors_portfolio[zero_factor_mask] = 1.0

        results_df["Portfolio Accumulated Gain Daily"] = (
            gain_factors_portfolio.cumprod()
        )
        if not results_df.empty and pd.isna(results_df["daily_return"].iloc[0]):
            results_df.iloc[
                0, results_df.columns.get_loc("Portfolio Accumulated Gain Daily")
            ] = np.nan

        # --- MODIFIED: Calculate final_twr_factor ONLY for the requested range ---
        if not results_df.empty:
            ts_start = pd.Timestamp(start_date_filter).tz_localize(results_df.index.tz)
            ts_end = pd.Timestamp(end_date_filter).tz_localize(
                results_df.index.tz
            ) + pd.Timedelta(hours=23, minutes=59)

            # Slice the gain factors for the requested period
            period_gain_factors = gain_factors_portfolio.loc[ts_start:ts_end]
            if not period_gain_factors.empty:
                final_twr_factor = period_gain_factors.prod()
            else:
                # Fallback to the last point if slicing yielded nothing (shouldn't happen with valid data)
                final_twr_factor = (
                    results_df["Portfolio Accumulated Gain Daily"].dropna().iloc[-1]
                    if not results_df.empty
                    else np.nan
                )

        for bm_symbol in benchmark_symbols_yf:
            price_col = f"{bm_symbol} Price"
            accum_col_daily = f"{bm_symbol} Accumulated Gain Daily"
            if price_col in results_df.columns:
                bench_prices_no_na = results_df[price_col].dropna()
                if not bench_prices_no_na.empty:
                    bench_daily_returns = (
                        bench_prices_no_na.pct_change(fill_method=None)
                        .reindex(results_df.index)
                        .ffill()
                        .fillna(0.0)
                    )
                    gain_factors_bench = 1 + bench_daily_returns
                    accum_gains_bench = gain_factors_bench.cumprod()
                    results_df[accum_col_daily] = accum_gains_bench
                    if not results_df.empty:
                        results_df.iloc[
                            0, results_df.columns.get_loc(accum_col_daily)
                        ] = np.nan
                else:
                    results_df[accum_col_daily] = np.nan
            else:
                results_df[accum_col_daily] = np.nan
        status_update += " Daily accum gain calc complete."

        # --- NEW: Calculate Absolute Gain / ROI on Daily Basis ---
        # This provides a money-weighted perspective for all views.
        if "value" in results_df.columns and "net_flow" in results_df.columns:
            results_df["Cumulative Net Flow"] = (
                results_df["net_flow"].fillna(0.0).cumsum()
            )
            results_df["Absolute Gain ($)"] = (
                results_df["value"] - results_df["Cumulative Net Flow"]
            )
            denom_roi = results_df["Cumulative Net Flow"]
            # Set ROI to NaN if Cost Basis is <= 0 to avoid massive misleading spikes
            # or undefined math (infinite return).
            roi_series = (results_df["Absolute Gain ($)"] / denom_roi) * 100.0
            roi_series[denom_roi <= 0] = np.nan
            results_df["Absolute ROI (%)"] = roi_series

        # --- NEW: Calculate Drawdown from Portfolio Accumulated Gain Daily ---
        if "Portfolio Accumulated Gain Daily" in results_df.columns:
            # 1. Calculate Running Max
            running_max = results_df["Portfolio Accumulated Gain Daily"].cummax()
            # 2. Drawdown = (Current / Running Max) - 1
            # Handle potential NaNs or zeros if needed (though cummax usually safe)
            results_df["drawdown"] = (
                results_df["Portfolio Accumulated Gain Daily"] / running_max
            ) - 1
            # 3. Convert to percentage? Frontend expects percentage number (e.g. -5.5 for -5.5%)?
            # Looking at PerformanceGraph.tsx, it renders `dataPoint.drawdown.toFixed(2) + '%'`.
            # If the value is -0.05 (for -5%), then * 100
            results_df["drawdown"] = results_df["drawdown"] * 100.0
            # Fill NaNs with 0.0
            results_df["drawdown"] = results_df["drawdown"].fillna(0.0)

        if interval in ["D", "1d"]:
            final_df_resampled = results_df
            final_df_resampled.rename(
                columns={
                    "Portfolio Accumulated Gain Daily": "Portfolio Accumulated Gain"
                },
                inplace=True,
            )
            for bm_symbol in benchmark_symbols_yf:
                accum_col_daily = f"{bm_symbol} Accumulated Gain Daily"
                accum_col_final = f"{bm_symbol} Accumulated Gain"
                if accum_col_daily in final_df_resampled.columns:
                    final_df_resampled.rename(
                        columns={accum_col_daily: accum_col_final}, inplace=True
                    )

            # --- FIX: Shift timestamp to Market Close (16:00 EST) ---
            # Standard YF data is 00:00 UTC, which equals 19:00 EST of the PREVIOUS day.
            # This causes the frontend to display Tuesday's data as Monday.
            # We shift it to 16:00 EST (21:00 UTC) so it falls correctly on the trading day.
            if not final_df_resampled.empty:
                try:
                    dates = final_df_resampled.index.date
                    naive_dti = pd.DatetimeIndex(dates)
                    ny_midnight = naive_dti.tz_localize("America/New_York")
                    ny_close = ny_midnight + pd.Timedelta(hours=16)
                    final_df_resampled.index = ny_close.tz_convert("UTC")
                    logging.info(
                        "Shifted Daily index to 16:00 EST to correct visual date alignment."
                    )
                except Exception as e_shift:
                    logging.error(f"Failed to shift daily index: {e_shift}")

        elif (
            interval in ["W", "M", "ME"] and not results_df.empty
        ):  # <-- ADD 'M' to the check
            # logging.info(f"Hist Final: Resampling to interval '{interval}'...")
            try:
                # --- ADDED: Log before resampling ---
                logging.debug(
                    f"Resampling '{interval}': Input results_df shape: {results_df.shape}"
                )
                # --- FIX: Determine correct resampling frequency ---
                resample_freq = interval
                if interval == "M":
                    resample_freq = "ME"  # Use Month End for 'M' interval
                # --- END FIX ---

                # --- MODIFIED: Include absolute investment tracking ---
                resampling_agg = {
                    "value": "last",
                    "daily_gain": "sum",
                    "net_flow": "sum",
                    "Portfolio Accumulated Gain Daily": "last",
                    "Cumulative Net Flow": "last",
                    "Absolute Gain ($)": "last",
                    "drawdown": "last",  # Preserve drawdown
                }
                for bm_symbol in benchmark_symbols_yf:
                    price_col = f"{bm_symbol} Price"
                    accum_col_daily = f"{bm_symbol} Accumulated Gain Daily"
                    if price_col in results_df.columns:
                        resampling_agg[price_col] = "last"
                    if accum_col_daily in results_df.columns:
                        resampling_agg[accum_col_daily] = "last"

                final_df_resampled = results_df.resample(resample_freq).agg(
                    resampling_agg
                )

                # --- ADDED: Log after resampling ---
                logging.debug(
                    f"Resampling '{interval}': Output final_df_resampled shape: {final_df_resampled.shape}"
                )

                # --- FIX: Use Daily Accumulated Gain directly ---
                # This ensures TWR is consistent across all timeframes.
                final_df_resampled.rename(
                    columns={
                        "Portfolio Accumulated Gain Daily": "Portfolio Accumulated Gain"
                    },
                    inplace=True,
                )
                for bm_symbol in benchmark_symbols_yf:
                    accum_col_daily = f"{bm_symbol} Accumulated Gain Daily"
                    accum_col_final = f"{bm_symbol} Accumulated Gain"
                    if accum_col_daily in final_df_resampled.columns:
                        final_df_resampled.rename(
                            columns={accum_col_daily: accum_col_final}, inplace=True
                        )

                # --- NEW: NO LONGER recalculating Absolute ROI here, it's summed/lasted from daily ---
                # But we do need to handle ROI on resampled data because simply taking 'last' ROI
                # or summing ROI doesn't make sense.
                # However, Absolute Gain is additive (if using sum of gains) or 'last' (if using cumulative flow).
                # Actually, using 'last' for Absolute Gain and Cumulative Flow IS correct because they are status-at-time.
                # But let's re-calculate ROI on the resampled values to be perfectly accurate.
                denom_roi_res = final_df_resampled["Cumulative Net Flow"].replace(
                    0, np.nan
                )
                final_df_resampled["Absolute ROI (%)"] = (
                    final_df_resampled["Absolute Gain ($)"] / denom_roi_res.abs()
                ) * 100.0

                status_update += f" Resampled to '{interval}' with Absolute metrics."
            except Exception as e_resample:
                logging.warning(
                    f"Hist WARN: Failed resampling to interval '{interval}': {e_resample}. Returning daily results."
                )
                status_update += f" Resampling failed ('{interval}')."
                final_df_resampled = results_df
        else:
            final_df_resampled = results_df
            final_df_resampled.rename(
                columns={
                    "Portfolio Accumulated Gain Daily": "Portfolio Accumulated Gain"
                },
                inplace=True,
            )
            for bm_symbol in benchmark_symbols_yf:
                accum_col_daily = f"{bm_symbol} Accumulated Gain Daily"
                accum_col_final = f"{bm_symbol} Accumulated Gain"
                if accum_col_daily in final_df_resampled.columns:
                    final_df_resampled.rename(
                        columns={accum_col_daily: accum_col_final}, inplace=True
                    )

        columns_to_keep = [
            "value",
            "daily_gain",
            "daily_return",
            "Portfolio Accumulated Gain",
            "Absolute Gain ($)",
            "Absolute ROI (%)",
            "Cumulative Net Flow",
            "drawdown",
        ]
        for bm_symbol in benchmark_symbols_yf:
            price_col = f"{bm_symbol} Price"
            accum_col_final = f"{bm_symbol} Accumulated Gain"
            if price_col in final_df_resampled.columns:
                columns_to_keep.append(price_col)
            if accum_col_final in final_df_resampled.columns:
                columns_to_keep.append(accum_col_final)

        # Ensure we only keep columns that actually exist
        columns_to_keep = [
            col for col in columns_to_keep if col in final_df_resampled.columns
        ]

        final_df_output = final_df_resampled[columns_to_keep].copy()
        final_df_output.rename(
            columns={"value": "Portfolio Value", "daily_gain": "Portfolio Daily Gain"},
            inplace=True,
        )
        # --- ADDED: Log final output DataFrame shape and tail ---
        logging.debug(
            f"[_calculate_accumulated_gains_and_resample] Final final_df_output shape: {final_df_output.shape}"
        )
        logging.debug(
            f"Final final_df_output tail:\n{final_df_output.tail().to_string()}"
        )
        if interval != "D" and "daily_return" in final_df_output.columns:
            final_df_output.drop(columns=["daily_return"], inplace=True)

        # --- ADDED: Filter final output based on requested date range ---
        if start_date_filter and end_date_filter:
            try:
                pd_start = pd.Timestamp(start_date_filter)
                pd_end = pd.Timestamp(end_date_filter) + pd.Timedelta(
                    hours=23, minutes=59, seconds=59
                )

                # Ensure index is timezone-naive before comparison if needed
                if final_df_output.index.tz is not None:
                    final_df_output.index = final_df_output.index.tz_localize(None)
                # --- NEW BASELINE LOGIC ($t_{-1}$) ---
                # Instead of normalizing by the first visible point (t0),
                # we want to normalize by the point immediately preceding it (t-1).
                # This ensures the first point (t0) already shows the return of that first day.

                # 1. Prepare a TZ-naive copy of the full resampled data for lookup
                resampled_naive = final_df_resampled.copy()
                if resampled_naive.index.tz is not None:
                    resampled_naive.index = resampled_naive.index.tz_localize(None)

                # 2. Identify first date in the visible range
                available_dates = final_df_output.index.sort_values()
                visible_mask = (available_dates >= pd_start) & (
                    available_dates <= pd_end
                )
                visible_dates = available_dates[visible_mask]

                if not visible_dates.empty:
                    # 3. Find the normalization baseline
                    for col in final_df_output.columns:
                        if "Accumulated Gain" in col:
                            divisor = 0.0
                            found_valid = False

                            # --- QUICK CHECK: If entire column is NaN or zero, skip search ---
                            col_data = final_df_output[col]
                            if col_data.isna().all() or (col_data == 0).all():
                                logging.debug(
                                    f"Normalization skipped for {col}: Column is empty or all zeros."
                                )
                                continue

                            # BUG-10 FIX: Try t-1 (last point BEFORE visible range) from full data
                            if col in resampled_naive.columns:
                                pre_range_data = resampled_naive.loc[
                                    resampled_naive.index < visible_dates[0], col
                                ].dropna()
                                if not pre_range_data.empty:
                                    t_minus_1_val = pre_range_data.iloc[-1]
                                    if pd.notnull(t_minus_1_val) and t_minus_1_val != 0:
                                        divisor = t_minus_1_val
                                        found_valid = True
                                        logging.debug(
                                            f"BUG-10 FIX: Using t-1 baseline for {col}: {divisor}"
                                        )

                            # Fallback: t0 if t-1 not available
                            if not found_valid:
                                t0_val = final_df_output.loc[visible_dates[0], col]
                                if pd.notnull(t0_val) and t0_val != 0:
                                    divisor = t0_val
                                    found_valid = True
                                    logging.debug(
                                        f"BUG-10 FIX: Fallback to t0 baseline for {col}: {divisor}"
                                    )
                                else:
                                    # t0 is invalid, search forwards
                                    logging.debug(
                                        f"t0 baseline invalid for {col} (val={t0_val}). Searching forward..."
                                    )
                                    for d in visible_dates:
                                        val = final_df_output.loc[d, col]
                                        if pd.notnull(val) and val != 0:
                                            divisor = val
                                            found_valid = True
                                            logging.debug(
                                                f"Found valid baseline for {col} at {d}: {divisor}"
                                            )
                                            break

                            if found_valid and divisor != 0:
                                # --- FIX: Safety check for massive spikes due to near-zero divisor ---
                                # If the divisor is extremely small (e.g. 0.0001), it will amplify the
                                # whole series by 10,000x. We set a threshold for validity.
                                if abs(divisor) < 1e-4:
                                    logging.warning(
                                        f"Normalization divisor for {col} is too small ({divisor}). Skipping normalization to prevent spikes."
                                    )
                                else:
                                    final_df_output[col] = (
                                        final_df_output[col] / divisor
                                    )
                            else:
                                logging.warning(
                                    f"Could not find ANY valid normalization baseline for {col} in range."
                                )

                # finally slice the output
                final_df_output = final_df_output.loc[pd_start:pd_end]

                logging.debug(
                    f"Filtered and normalized final output to range: {start_date_filter} - {end_date_filter}"
                )
            except Exception as e_final_filter:
                logging.warning(f"Could not apply final date filter: {e_final_filter}")
        # --- END ADDED ---
    except Exception:
        logging.exception("Hist CRITICAL: Accum gain/resample calc error")
        status_update += " Accum gain/resample calc failed."
        return pd.DataFrame(), np.nan, status_update

    return final_df_output, final_twr_factor, status_update
