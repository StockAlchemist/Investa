"""Analytics routes: capital gains, dividends, income, risk, attribution, calendar."""

# ruff: noqa: E402
import logging
from collections import defaultdict
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Set

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

import config
from finutils import is_cash_symbol
from market_data import map_to_yf_symbol
from portfolio_analyzer import (
    extract_dividend_history,
    extract_realized_capital_gains_history,
    generate_cash_interest_events,
)
from risk_metrics import calculate_all_risk_metrics
from server.auth import User
from server.dependencies import get_config_manager, get_current_user, get_transaction_data
from server.portfolio_service import (
    _calculate_portfolio_summary_internal,
    _get_historical_performance_cached,
)
from server.route_utils import clean_nans, get_mdp

router = APIRouter()


@router.get("/capital_gains")
async def get_capital_gains(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    from_date: Optional[str] = Query(None, alias="from"),
    to_date: Optional[str] = Query(None, alias="to"),
    data: tuple = Depends(get_transaction_data)
):
    """
    Returns the realized capital gains history.

    Args:
        currency (str): The display currency.
        accounts (List[str]): List of account names.
        data (tuple): Dependency injection.

    Returns:
        List[Dict]: A list of realized gain/loss records.
    """
    (
        df,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        account_cash_mode_map,
        original_csv_path,
        db_mtime
    ) = data

    if df.empty:
        return []

    try:
        # Determine full date range from transactions
        min_date = df["Date"].min().date()
        max_date = date.today()
        
        _, _, historical_fx_yf, _ = await _get_historical_performance_cached(
            df=df,
            manual_overrides_dict=manual_overrides,
            user_symbol_map=user_symbol_map,
            user_excluded_symbols=user_excluded_symbols,
            account_currency_map=account_currency_map,
            original_csv_file_path=original_csv_path,
            start_date=min_date,
            end_date=max_date,
            interval="D",
            benchmark_symbols_yf=[], # No benchmarks needed
            display_currency=currency,
            include_accounts=accounts,
            account_cash_mode_map=account_cash_mode_map,
            db_mtime=db_mtime
        )
        
        # Parse dates
        start_dt = None
        end_dt = None
        if from_date:
            try:
                start_dt = date.fromisoformat(from_date)
            except ValueError:
                pass
        if to_date:
            try:
                end_dt = date.fromisoformat(to_date)
            except ValueError:
                pass

        capital_gains_df = extract_realized_capital_gains_history(
            all_transactions_df=df,
            display_currency=currency,
            historical_fx_yf=historical_fx_yf,
            default_currency=config.DEFAULT_CURRENCY,
            shortable_symbols=config.SHORTABLE_SYMBOLS,
            include_accounts=accounts,
            from_date=start_dt,
            to_date=end_dt
        )
        
        if capital_gains_df.empty:
            return []
            
        # Convert to list of dicts and clean NaNs
        # Ensure Date is string
        if 'Date' in capital_gains_df.columns:
            capital_gains_df['Date'] = capital_gains_df['Date'].astype(str)
            
        records = capital_gains_df.to_dict(orient="records")
        return clean_nans(records)

    except Exception as e:
        logging.error(f"Error getting capital gains: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dividends")
async def get_dividends(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data)
):
    """
    Returns the dividend history.

    Args:
        currency (str): The display currency.
        accounts (List[str], optional): List of account names.
        data (tuple): Dependency injection.

    Returns:
        List[Dict]: A list of dividend records.
    """
    (
        df,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        account_cash_mode_map,
        original_csv_path,
        db_mtime
    ) = data

    if df.empty:
        return []

    try:
        # Determine full date range from transactions
        min_date = df["Date"].min().date()
        max_date = date.today()
        
        _, _, historical_fx_yf, _ = await _get_historical_performance_cached(
            df=df,
            manual_overrides_dict=manual_overrides,
            user_symbol_map=user_symbol_map,
            user_excluded_symbols=user_excluded_symbols,
            account_currency_map=account_currency_map,
            original_csv_file_path=original_csv_path,
            start_date=min_date,
            end_date=max_date,
            interval="D",
            benchmark_symbols_yf=[], # No benchmarks needed
            display_currency=currency,
            include_accounts=accounts,
            account_cash_mode_map=account_cash_mode_map,
            db_mtime=db_mtime
        )
        
        dividend_df = extract_dividend_history(
            all_transactions_df=df,
            display_currency=currency,
            historical_fx_yf=historical_fx_yf,
            default_currency=config.DEFAULT_CURRENCY,
            include_accounts=accounts
        )
        
        if dividend_df.empty:
            return []
            
        # Convert to list of dicts and clean NaNs
        # Ensure Date is string
        if 'Date' in dividend_df.columns:
            dividend_df['Date'] = dividend_df['Date'].astype(str)
            
        records = dividend_df.to_dict(orient="records")
        return clean_nans(records)

    except Exception as e:
        logging.error(f"Error getting dividends: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _generate_dividend_events(
    holdings: Dict[str, float],
    user_symbol_map: Dict,
    user_excluded_symbols: Set,
    current_user: User,
    portfolio_summary_rows: List[dict]
) -> List[dict]:
    """
    Shared logic to generate dividend events (confirmed + estimated) + cash interest
    for the next 12 months.
    """
    from market_data import _run_isolated_fetch
    from finutils import get_dividend_details
    import concurrent.futures
    
    provider = get_mdp()
    yf_map = {} # Internal -> YF
    
    # Map symbols
    for sym in holdings.keys():
        if is_cash_symbol(sym):
            continue
        yf_sym = map_to_yf_symbol(sym, user_symbol_map, user_excluded_symbols)
        if yf_sym:
            yf_map[sym] = yf_sym

    calendar_events = []
    today = date.today()
    end_date = today + timedelta(days=365) # 1 Year Projection
    
    def fetch_symbol_data(sym):
        """Helper to fetch data for a single symbol independently."""
        yf_sym = yf_map.get(sym)
        if not yf_sym:
            return []
        
        local_events = []
        qty = holdings.get(sym, 0)
        
        try:
            # Use MarketDataProvider cache for fundamentals (lighter and cached)
            info = provider.get_fundamental_data(yf_sym) or {}

            # Forward-looking dividend rate & cadence (prefers Yahoo's
            # `dividendRate`, so a just-announced increase/cut is reflected even
            # before `lastDividendValue` catches up — see get_dividend_details).
            details = get_dividend_details(info)
            indicated_rate = details["indicated_annual_rate"]
            freq_months = details["frequency_months"]
            per_period_amt = (
                indicated_rate / (12 // freq_months) if freq_months else indicated_rate
            )

            # 1. Confirmed Events (Need live ticker for calendar, but wrap carefully)
            # Only check calendar if we have indication of dividends
            if indicated_rate > 0:
                # Use isolated fetch for calendar data
                cal = _run_isolated_fetch([yf_sym], task="calendar")
                if cal and 'Dividend Date' in cal:
                    div_date_raw = cal['Dividend Date']
                    # Dates were stringified in the worker
                    c_date = None
                    if isinstance(div_date_raw, str):
                        try:
                            c_date = datetime.fromisoformat(div_date_raw).date()
                        except ValueError:
                            # Fallback if it's just 'YYYY-MM-DD'
                            try:
                                c_date = datetime.strptime(div_date_raw, "%Y-%m-%d").date()
                            except ValueError:
                                pass
                    else:
                        c_date = div_date_raw
                        if isinstance(c_date, datetime):
                            c_date = c_date.date()
                    
                    if c_date and c_date >= today:
                        local_events.append({
                            "symbol": sym,
                            "dividend_date": str(c_date),
                            "ex_dividend_date": str(cal.get('Ex-Dividend Date', '')),
                            "amount": per_period_amt * qty,
                            "status": "confirmed"
                        })

            # 2. Estimated Events
            if indicated_rate > 0:

                
                # Anchor
                anchor = None
                if local_events:
                    try:
                        anchor = datetime.strptime(local_events[-1]["dividend_date"], "%Y-%m-%d").date()
                    except (ValueError, KeyError, IndexError) as e:
                        logging.debug(f"Dividend anchor parse (local_events): {e}")

                if not anchor and info.get("lastDividendDate"):
                    try:
                        anchor = date.fromtimestamp(info.get("lastDividendDate"))
                    except (OSError, ValueError, TypeError) as e:
                        logging.debug(f"Dividend anchor parse (lastDividendDate): {e}")

                if not anchor and info.get("exDividendDate"):
                    try:
                        anchor = date.fromtimestamp(info.get("exDividendDate")) + timedelta(days=21)
                    except (OSError, ValueError, TypeError) as e:
                        logging.debug(f"Dividend anchor parse (exDividendDate): {e}")
                    
                if anchor:
                    curr = anchor
                    while curr < today:
                        curr = (pd.Timestamp(curr) + pd.DateOffset(months=freq_months)).date()
                    
                    while curr <= end_date:
                        is_dup = False
                        for ce in local_events:
                            if ce["status"] == "confirmed":
                                ce_date = datetime.strptime(ce["dividend_date"], "%Y-%m-%d").date()
                                if abs((ce_date - curr).days) < 20: 
                                    is_dup = True
                                    break
                        
                        if not is_dup and curr >= today:
                            est_amt = (indicated_rate / (12 // freq_months)) * qty
                            local_events.append({
                                "symbol": sym,
                                "dividend_date": str(curr),
                                "ex_dividend_date": "",
                                "amount": est_amt,
                                "status": "estimated"
                            })
                        
                        curr = (pd.Timestamp(curr) + pd.DateOffset(months=freq_months)).date()
                        
        except Exception as e:
            logging.warning(f"Error fetching data for {sym}: {e}")
            
        return local_events

    # Run in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_sym = {executor.submit(fetch_symbol_data, sym): sym for sym in holdings.keys()}
        for future in concurrent.futures.as_completed(future_to_sym):
            try:
                events = future.result()
                calendar_events.extend(events)
            except Exception as exc:
                logging.error(f"Symbol generated an exception: {exc}")

    # Sort by date
    calendar_events.sort(key=lambda x: x["dividend_date"])
    
    try:
        # Load settings for cash interest using the GLOBAL config manager if possible, or create new?
        # Creating new ConfigManager might be expensive or thread-unsafe if not careful.
        # But we are in async context.
        # Ideally passed in, but for now we look it up.
        config_manager = get_config_manager(current_user)
        config_manager.load_manual_overrides()
        interest_rates = config_manager.manual_overrides.get("account_interest_rates", {})
        thresholds = config_manager.manual_overrides.get("interest_free_thresholds", {})
        
        # Detect currency
        display_currency = "USD"
        if portfolio_summary_rows:
            for k in portfolio_summary_rows[0].keys():
                if k.startswith("Market Value ("):
                        parts = k.split("(")
                        if len(parts) > 1:
                            display_currency = parts[1].split(")")[0]
                        break

        try:
            cash_events = generate_cash_interest_events(
                portfolio_summary_rows=portfolio_summary_rows,
                interest_rates=interest_rates,
                thresholds=thresholds,
                start_date=today,
                end_date=end_date,
                display_currency=display_currency
            )
            
            if cash_events:
                 logging.info(f"Generated {len(cash_events)} cash interest events for {current_user.username}")
                 calendar_events.extend(cash_events)
                 
        except Exception as e:
             logging.error(f"Error generating cash interest events: {e}")

    except Exception as e:
        logging.error(f"Error processing cash interest/settings: {e}")
        
    return calendar_events


@router.get("/projected_income")
async def get_projected_income(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data),
    current_user: User = Depends(get_current_user)
):
    """
    Returns projected monthly dividend income for the next 12 months.
    """
    try:
        # 1. Get Current Holdings
        summary_data = await _calculate_portfolio_summary_internal(
            currency=currency,
            include_accounts=accounts,
            show_closed_positions=False,
            data=data,
            current_user=current_user
        )
        summary_df = summary_data.get("summary_df")
        if summary_df is None or summary_df.empty:
            return []
            
        # 2. Extract Holdings & Quantities
        holdings = defaultdict(float) 
        rows = summary_df.to_dict(orient="records")
        for row in rows:
            sym = row.get("Symbol")
            if sym == "Total" or row.get("is_total"):
                continue
            qty = row.get("Quantity", 0)
            if qty > 0:
                holdings[sym] += qty
                
        if not holdings:
            return []

        # 3. Use unified event generation logic
        df, _, user_symbol_map, user_excluded_symbols, _, _, _, _ = data
        
        # Calculate raw events using the robust logic (Calendar + Estimates + Cash Interest)
        events = await _generate_dividend_events(
            holdings=holdings,
            user_symbol_map=user_symbol_map,
            user_excluded_symbols=user_excluded_symbols,
            current_user=current_user,
            portfolio_summary_rows=rows
        )
        
        # 4. Aggregate into Monthly Buckets (with Currency Conversion)
        # We need to convert LOCAL currency dividend amounts to the DISPLAY currency.
        # We use the FX rates fetched during the summary calculation.
        from finutils import get_conversion_rate
        fx_rates_vs_usd = summary_data["metrics"].get("_fx_rates_vs_usd", {})
        
        # Build a mapping of Symbol -> FX Rate (to display currency)
        symbol_to_fx_rate = {}
        for row in rows:
            row_sym = row.get("Symbol")
            local_curr = row.get("Local Currency")
            if row_sym and local_curr:
                rate = get_conversion_rate(local_curr, currency, fx_rates_vs_usd)
                if pd.notna(rate):
                    symbol_to_fx_rate[row_sym] = rate
        
        # Structure: "YYYY-MM" -> {"total": float, "breakdown": {symbol: float}}
        projection = defaultdict(lambda: {"total": 0.0, "breakdown": defaultdict(float)})
        
        today = date.today()
        # We want to cover the next 12 months
        
        for event in events:
            # Event has: symbol, dividend_date (str YYYY-MM-DD), amount, status
            try:
                evt_date = datetime.strptime(event["dividend_date"], "%Y-%m-%d").date()
                key = evt_date.strftime("%Y-%m")
                sym = event["symbol"]
                amt = event["amount"] # This is usually LOCAL amount (except for Cash Interest)
                
                # Apply FX conversion if we have a rate for this symbol
                # Only symbols in symbol_to_fx_rate need conversion (Cash Interest events use $CASH which isn't in map)
                if sym in symbol_to_fx_rate:
                    amt *= symbol_to_fx_rate[sym]
                
                # Only include if within 12 months (events list is already limited to ~1 year by generator, but double check)
                if evt_date >= today and evt_date <= today + timedelta(days=365):
                    projection[key]["total"] += amt
                    projection[key]["breakdown"][sym] += amt
            except Exception as e:
                logging.warning(f"Error aggregating event {event}: {e}")

        # 5. Format Result for Graph
        results = []
        iter_date = today.replace(day=1)
        for _ in range(12):
            key = iter_date.strftime("%Y-%m")
            label = iter_date.strftime("%b %Y")
            data_point = projection.get(key, {"total": 0.0, "breakdown": {}})
            
            # Ensure breakdown dict exists even if empty
            if not data_point.get("breakdown"):
                 data_point["breakdown"] = {}

            results.append({
                "month": label,
                "year_month": key,
                "value": data_point["total"],
                **data_point["breakdown"]
            })
            iter_date = (pd.Timestamp(iter_date) + pd.DateOffset(months=1)).date()
            
        return results
    except Exception as e:
        logging.error(f"Error getting projected income: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk_metrics")
async def get_risk_metrics(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data)
):
    """
    Returns portfolio risk metrics (Sharpe, Volatility, Max Drawdown).
    """
    df, manual_overrides, user_symbol_map, user_excluded_symbols, account_currency_map, account_cash_mode_map, original_csv_path, _ = data
    if df.empty:
        return {}

    try:
        # Calculate daily history to get the total portfolio value series
        daily_df, _, _, _ = await _get_historical_performance_cached(
            df=df,
            manual_overrides_dict=manual_overrides,
            user_symbol_map=user_symbol_map,
            user_excluded_symbols=user_excluded_symbols,
            account_currency_map=account_currency_map,
            original_csv_file_path=original_csv_path,
            start_date=date(2000, 1, 1),
            end_date=date.today(),
            display_currency=currency,
            include_accounts=accounts,
            benchmark_symbols_yf=['^GSPC'], # Add S&P 500 for risk metrics
            interval="D",
            account_cash_mode_map=account_cash_mode_map,
            db_mtime=data[7]  # db_mtime
        )
        
        if daily_df is None or "Portfolio Accumulated Gain" not in daily_df.columns:
            return {}

        # Use Portfolio Accumulated Gain (TWR series) for consistent risk metrics
        portfolio_values = daily_df["Portfolio Accumulated Gain"]
        benchmark_values = daily_df['^GSPC Price'] if '^GSPC Price' in daily_df.columns else None
        metrics = calculate_all_risk_metrics(portfolio_values, benchmark_values=benchmark_values)
        
        # --- FIX: Calculate YTD Return from daily_df ---
        try:
            if not daily_df.empty:
                import pandas as pd
                # Ensure index is datetime for filtering
                if not pd.api.types.is_datetime64_any_dtype(daily_df.index):
                    daily_df.index = pd.to_datetime(daily_df.index)
                
                current_year = date.today().year
                prev_year_data = daily_df[daily_df.index.year < current_year]
                
                if not prev_year_data.empty:
                    prev_year_twr = prev_year_data["Portfolio Accumulated Gain"].iloc[-1]
                    current_twr = portfolio_values.iloc[-1]
                    if prev_year_twr and prev_year_twr > 0:
                        metrics["YTD Return"] = (current_twr / prev_year_twr) - 1.0
                else:
                    # Account started this year
                    current_twr = portfolio_values.iloc[-1]
                    start_twr = portfolio_values.iloc[0]
                    if start_twr > 0:
                        metrics["YTD Return"] = (current_twr / start_twr) - 1.0
        except Exception as ytd_e:
            logging.error(f"Error calculating YTD Return in risk_metrics: {ytd_e}")

        if metrics.get('Beta') is None:
            logging.warning(f"Risk Metrics: Beta is None. daily_df columns: {daily_df.columns.tolist()}, port_len={len(portfolio_values)}, bench_len={len(benchmark_values) if benchmark_values is not None else 0}")
        return clean_nans(metrics)

    except Exception as e:
        logging.error(f"Error calculating risk metrics: {e}")
        return {"error": str(e)}


@router.get("/attribution")
async def get_attribution(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    show_all: bool = False,
    show_closed: Optional[bool] = Query(None),
    data: tuple = Depends(get_transaction_data),
    current_user: User = Depends(get_current_user)
):
    """
    Returns performance attribution by sector and stock.
    """
    df, manual_overrides, user_symbol_map, user_excluded_symbols, account_currency_map, account_cash_mode_map, original_csv_path, _ = data
    if df.empty:
        return {}

    try:
        # Get current summary rows which contain gains and sector info
        summary_data = await _calculate_portfolio_summary_internal(
            currency=currency,
            include_accounts=accounts,
            show_closed_positions=show_closed if show_closed is not None else True,
            data=data,
            current_user=current_user
        )
        
        summary_df = summary_data.get("summary_df")
        if summary_df is None or summary_df.empty:
            return {"sectors": [], "stocks": []}
        
        rows = summary_df.to_dict(orient="records")

        # Sector Attribution
        sector_data = defaultdict(lambda: {"gain": 0.0, "value": 0.0})
        stock_data = []

        total_gain = 0.0
        total_cost = 0.0
        for row in rows:
            symbol = row.get("Symbol")
            if symbol == "Total" or row.get("is_total"):
                continue
            
            gain_col = f"Total Gain ({currency})"
            cost_col = f"Total Cost Invested ({currency})"
            value_col = f"Market Value ({currency})"
            
            gain = row.get(gain_col, 0.0)
            cost = row.get(cost_col, 0.0)
            value = row.get(value_col, 0.0)
            sector = row.get("Sector") or "Unknown"
            
            total_gain += gain
            total_cost += cost
            sector_data[sector]["gain"] += gain
            sector_data[sector]["value"] += value
            
            # --- AGGREGATION LOGIC ---
            # Group by Name if available to merge tickers (e.g. GOOG/GOOGL), otherwise by symbol
            name = row.get("Name")
            found = False
            for prev_item in stock_data:
                # Check for match by Name (if valid) OR Symbol
                # match if names are identical strings (and not None/empty)
                name_match = (name and prev_item.get("name") and name == prev_item.get("name"))
                
                # Check if this exact symbol is already part of this group (could be a comma-separated list)
                existing_syms = [s.strip() for s in prev_item["symbol"].split(",")]
                symbol_match = (symbol in existing_syms)

                if name_match or symbol_match:
                     prev_item["gain"] += gain
                     prev_item["value"] += value
                     
                     # If it was a name match but the specific symbol wasn't listed yet, merge it
                     if not symbol_match:
                         prev_item["symbol"] += f", {symbol}"
                     
                     found = True
                     break
            
            if not found:
                 stock_data.append({
                    "symbol": symbol,
                    "name": name,
                    "gain": gain,
                    "value": value,
                    "sector": sector
                })


        # Format sector output
        sector_attribution = []
        for sector, d in sector_data.items():
            sector_attribution.append({
                "sector": sector,
                "gain": d["gain"],
                "value": d["value"],
                "contribution": (d["gain"] / total_cost if total_cost != 0 else 0)
            })

        # Sort by gain descending (winners first)
        sector_attribution.sort(key=lambda x: x.get("gain", 0), reverse=True)
        # Sort by gain descending (winners first)
        stock_data.sort(key=lambda x: x.get("gain", 0), reverse=True)
        
        # Calculate contribution % for stocks
        for stock in stock_data:
            stock["contribution"] = (stock["gain"] / total_cost) if total_cost != 0 else 0.0

        return clean_nans({
            "sectors": sector_attribution,
            "stocks": stock_data if show_all else stock_data[:10], # Top 10 contributors/detractors by default
            "total_gain": total_gain
        })
    except Exception as e:
        logging.error(f"Error calculating attribution: {e}")
        return {"error": str(e)}


def calculate_mtd_average_daily_balance(
    current_cash: float, 
    mtd_transactions: pd.DataFrame, 
    today_date: date
) -> float:
    """
    Calculates the Month-To-Date (MTD) Average Daily Balance (ADB).
    Reconstructs daily balances backwards from the current cash balance.
    """
    if current_cash == 0 and mtd_transactions.empty:
        return 0.0

    start_of_month = date(today_date.year, today_date.month, 1)
    days_in_month_so_far = (today_date - start_of_month).days + 1
    
    # Map dates to net change (assuming 'Total Amount' is signed correctly: + for inflow, - for outflow)
    # NOTE: In Investa, 'Total Amount' for BUY is negative (cash outflow), SELL is positive (cash inflow).
    # DEPOSIT is positive, WITHDRAWAL is negative.
    # So 'Total Amount' directly represents the change in cash balance.
    
    changes_by_date = {}
    if not mtd_transactions.empty:
        # Group by Date
        # Ensure Date column is datetime/date
        changes_by_date = mtd_transactions.groupby(mtd_transactions["Date"].dt.date)["Total Amount"].sum().to_dict()

    daily_balances = []
    
    # We walk BACKWARDS from Today
    running_balance = current_cash
    
    for d_idx in range(days_in_month_so_far):
        # Current day we are looking at (going backwards: Today, Yesterday, ...)
        lookback_date = today_date - timedelta(days=d_idx)
        
        # The running_balance represents the END OF DAY balance for lookback_date
        daily_balances.append(running_balance)
        
        # Before moving to yesterday (next iteration), adjust balance to get the start of day (end of previous day)
        # Start + Change = End  => Start = End - Change
        change_on_day = changes_by_date.get(lookback_date, 0.0)
        
        running_balance = running_balance - change_on_day
        
    if not daily_balances:
        return 0.0
        
    return sum(daily_balances) / len(daily_balances)


@router.get("/dividend_calendar")
async def get_dividend_calendar(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data),
    config_manager = Depends(get_config_manager),
    current_user: User = Depends(get_current_user)
):
    """
    Returns confirmed AND estimated dividend events for the next 12 months,
    with amounts converted to the requested display currency.
    """
    try:
        # 1. Get Current Holdings (in the display currency so the summary's FX
        # rates and cash-interest events are expressed in that currency).
        summary_data = await _calculate_portfolio_summary_internal(
            currency=currency,
            include_accounts=accounts,
            show_closed_positions=False,
            data=data,
            current_user=current_user
        )
        summary_df = summary_data.get("summary_df")
        if summary_df is None or summary_df.empty:
                return []
        rows = summary_df.to_dict(orient="records")

        # 2. Extract Holdings
        holdings = defaultdict(float)
        for r in rows:
            sym = r["Symbol"]
            if sym != "Total" and not r.get("is_total"):
                    qty = r.get("Quantity", 0)
                    if qty > 0:
                        holdings[sym] += qty

        if not holdings:
            return []

        # 3. Use unified event generation logic
        df, _, user_symbol_map, user_excluded_symbols, _, _, _, _ = data

        events = await _generate_dividend_events(
            holdings=holdings,
            user_symbol_map=user_symbol_map,
            user_excluded_symbols=user_excluded_symbols,
            current_user=current_user,
            portfolio_summary_rows=rows
        )

        # 4. Convert per-symbol LOCAL dividend amounts to the display currency
        # using the FX rates fetched during the summary calculation. Cash
        # interest events (symbol $CASH) are already generated in the display
        # currency, so they are intentionally absent from the rate map and left
        # unconverted.
        from finutils import get_conversion_rate
        fx_rates_vs_usd = summary_data["metrics"].get("_fx_rates_vs_usd", {})
        symbol_to_fx_rate = {}
        for row in rows:
            row_sym = row.get("Symbol")
            local_curr = row.get("Local Currency")
            if row_sym and local_curr:
                rate = get_conversion_rate(local_curr, currency, fx_rates_vs_usd)
                if pd.notna(rate):
                    symbol_to_fx_rate[row_sym] = rate

        for event in events:
            rate = symbol_to_fx_rate.get(event.get("symbol"))
            if rate is not None:
                event["amount"] *= rate

        return clean_nans(events)

    except Exception as e:
        logging.error(f"Error getting dividend calendar: {e}", exc_info=True)
        return []
