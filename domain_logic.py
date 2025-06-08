# domain_logic.py
# This file will contain domain-specific logic, data fetching, and calculation functions
# formerly in main_gui.py.

import pandas as pd
import numpy as np
import datetime
from datetime import date, timedelta, timezone
import logging
import requests
import yfinance as yf
import sqlite3
import re
import json
import traceback
import os
import sys

# Attempt to import config variables, with fallbacks if config.py is not found
try:
    from config import API_KEY, NEWS_API_KEY, DATABASE_PATH, CSV_FILE_PATH, LOG_FILE_PATH, RISK_FREE_RATE
except ImportError:
    API_KEY = os.getenv("API_KEY")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    DATABASE_PATH = "portfolio_database.db"
    CSV_FILE_PATH = "transactions.csv"
    LOG_FILE_PATH = "portfolio_app.log"
    RISK_FREE_RATE = 0.02  # Default risk-free rate

# Setup logging (consider moving to a central logging configuration if needed)
# For now, keep it simple; main_gui.py likely already configures the root logger.
logger = logging.getLogger(__name__)

# Placeholder for functions to be moved
# Example:
# def calculate_portfolio_performance(transactions_df, prices_df, fx_rates_df, cash_flows_df):
#     # ... implementation ...
#     pass


def _calculate_annualized_twr(total_twr_factor, start_date, end_date):
    """
    Calculates the annualized Time-Weighted Return (TWR) percentage.

    Args:
        total_twr_factor (float | np.nan): The total TWR factor (1 + total TWR)
                                           over the period.
        start_date (date | None): The start date of the period.
        end_date (date | None): The end date of the period.

    Returns:
        float | np.nan: The annualized TWR as a percentage, or np.nan if inputs
                        are invalid or calculation fails.
    """
    if pd.isna(total_twr_factor) or total_twr_factor <= 0:
        return np.nan
    if start_date is None or end_date is None or start_date >= end_date:
        return np.nan
    try:
        num_days = (end_date - start_date).days
        if num_days <= 0:
            return np.nan
        annualized_twr_factor = total_twr_factor ** (365.25 / num_days)
        return (annualized_twr_factor - 1) * 100.0
    except (TypeError, ValueError, OverflowError):
        return np.nan


if __name__ == '__main__':
    # This section can be used for testing functions in this module independently
    logger.info("domain_logic.py executed directly (for testing or utilities).")
    # Example: test a function here
    # test_transactions = pd.DataFrame(...)
    # test_prices = pd.DataFrame(...)
    # performance = calculate_portfolio_performance(test_transactions, test_prices, ...)
    # print(performance)
