# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          data_store.py
 Purpose:       Centralized data model for the Investa application.
                This class holds all portfolio-related data, such as holdings,
                summary metrics, historical data, etc., to decouple the main
                application logic from the data itself.

 Author:        Jules
 Created:       03/09/2025
 Copyright:     (c) Jules 2025
 Licence:       MIT
-------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
from PySide6.QtCore import QObject, Signal
from typing import Dict, Any, List, Set, Optional

class DataStore(QObject):
    """
    A centralized data store for the portfolio application.
    It holds all data and emits signals when the data is updated.
    """
    # Signals to notify the UI about data changes
    data_updated = Signal()
    holdings_updated = Signal(pd.DataFrame)
    summary_updated = Signal(dict)
    historical_updated = Signal(pd.DataFrame)
    # Add more specific signals as needed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.reset()

    def reset(self):
        """Resets all data to its initial empty state."""
        self.all_transactions_df_cleaned_for_logic: pd.DataFrame = pd.DataFrame()
        self.original_data: pd.DataFrame = pd.DataFrame()
        self.original_transactions_df_for_ignored_context: pd.DataFrame = pd.DataFrame()
        self.original_to_cleaned_header_map_from_csv: Dict[str, str] = {}
        self.holdings_data: pd.DataFrame = pd.DataFrame()
        self.summary_metrics_data: Dict[str, Any] = {}
        self.account_metrics_data: Dict[str, Any] = {}
        self.historical_data: pd.DataFrame = pd.DataFrame()
        self.full_historical_data: pd.DataFrame = pd.DataFrame()
        self.periodic_returns_data: Dict[str, pd.DataFrame] = {}
        self.periodic_value_changes_data: Dict[str, pd.DataFrame] = {}
        self.dividend_history_data: pd.DataFrame = pd.DataFrame()
        self.capital_gains_history_data: pd.DataFrame = pd.DataFrame()
        self.correlation_matrix_df: pd.DataFrame = pd.DataFrame()
        self.factor_analysis_results: Optional[Dict] = None
        self.scenario_analysis_result: Dict = {}
        self.ignored_data: pd.DataFrame = pd.DataFrame()
        self.index_quote_data: Dict[str, Dict[str, Any]] = {}
        self.internal_to_yf_map: Dict[str, str] = {}
        self.historical_prices_yf_adjusted: Dict[str, pd.DataFrame] = {}
        self.historical_fx_yf: Dict[str, pd.DataFrame] = {}
        self.available_accounts: List[str] = []
        self.last_calc_status: str = ""
        self.last_hist_twr_factor: float = np.nan

    def update_data(self, results: Dict[str, Any]):
        """
        Updates the data store with a dictionary of new data.

        Args:
            results (Dict[str, Any]): A dictionary containing the new data.
                                      Keys should match the attributes of this class.
        """
        self.summary_metrics_data = results.get('summary_metrics', {})
        self.holdings_data = results.get('holdings_df', pd.DataFrame())
        self.account_metrics_data = results.get('account_metrics', {})
        self.full_historical_data = results.get('full_historical_data_df', pd.DataFrame())
        self.historical_prices_yf_adjusted = results.get('hist_prices_adj', {})
        self.historical_fx_yf = results.get('hist_fx', {})
        self.ignored_data = self._process_ignored_data(
            results.get('combined_ignored_indices', set()),
            results.get('combined_ignored_reasons', {}),
            results.get('original_transactions_df_for_ignored', pd.DataFrame())
        )
        self.dividend_history_data = results.get('dividend_history_df', pd.DataFrame())
        self.capital_gains_history_data = results.get('capital_gains_history_df', pd.DataFrame())
        self.correlation_matrix_df = results.get('correlation_matrix_df', pd.DataFrame())
        self.factor_analysis_results = results.get('factor_analysis_results', {})
        self.scenario_analysis_result = results.get('scenario_analysis_result', {})
        self.index_quote_data = results.get('index_quotes', {})
        self.last_calc_status = results.get('status_msg', "Status Unknown")

        if 'Portfolio Accumulated Gain' in self.full_historical_data.columns:
            accum_gain_series = self.full_historical_data['Portfolio Accumulated Gain'].dropna()
            if len(accum_gain_series) >= 2:
                start_gain = accum_gain_series.iloc[0]
                end_gain = accum_gain_series.iloc[-1]
                if pd.notna(start_gain) and pd.notna(end_gain) and start_gain != 0:
                    self.last_hist_twr_factor = end_gain / start_gain

        self.data_updated.emit()

    def _process_ignored_data(self, ignored_indices: Set[int], ignored_reasons: Dict[int, str], original_df: pd.DataFrame) -> pd.DataFrame:
        """Processes ignored data to create a displayable DataFrame."""
        if not ignored_indices or original_df.empty:
            return pd.DataFrame()
        try:
            if "original_index" not in original_df.columns:
                original_df['original_index'] = original_df.index

            indices_to_check = {int(i) for i in ignored_indices if pd.notna(i)}
            valid_indices_mask = original_df["original_index"].isin(indices_to_check)
            ignored_rows_df = original_df[valid_indices_mask].copy()
            if not ignored_rows_df.empty:
                reasons_mapped = ignored_rows_df["original_index"].map(ignored_reasons).fillna("Unknown Reason")
                ignored_rows_df["Reason Ignored"] = reasons_mapped
                return ignored_rows_df.sort_values(by="original_index")
        except Exception as e:
            logging.error(f"Error processing ignored data: {e}", exc_info=True)
        return pd.DataFrame()

    def clear_all_data(self):
        """Resets all data to its initial empty state."""
        self.reset()
        self.data_updated.emit()

    def get_summary_metrics(self) -> Dict[str, Any]:
        return self.summary_metrics_data

    def get_holdings(self) -> pd.DataFrame:
        return self.holdings_data

    def get_all_transactions_df(self) -> pd.DataFrame:
        return self.all_transactions_df_cleaned_for_logic

    def set_all_transactions_df(self, df: pd.DataFrame):
        self.all_transactions_df_cleaned_for_logic = df

    def get_original_data(self) -> pd.DataFrame:
        return self.original_data

    def set_original_data(self, df: pd.DataFrame):
        self.original_data = df

    def get_original_transactions_df_for_ignored_context(self) -> pd.DataFrame:
        return self.original_transactions_df_for_ignored_context

    def set_original_transactions_df_for_ignored_context(self, df: pd.DataFrame):
        self.original_transactions_df_for_ignored_context = df

    def get_original_to_cleaned_header_map_from_csv(self) -> Dict[str, str]:
        return self.original_to_cleaned_header_map_from_csv

    def set_original_to_cleaned_header_map_from_csv(self, mapping: Dict[str, str]):
        self.original_to_cleaned_header_map_from_csv = mapping

    def get_last_calc_status(self) -> str:
        return self.last_calc_status

    def set_last_calc_status(self, status: str):
        self.last_calc_status = status

    def get_last_hist_twr_factor(self) -> float:
        return self.last_hist_twr_factor

    def set_last_hist_twr_factor(self, factor: float):
        self.last_hist_twr_factor = factor

    def get_holdings_data(self) -> pd.DataFrame:
        return self.holdings_data

    def set_holdings_data(self, df: pd.DataFrame):
        self.holdings_data = df

    def get_summary_metrics_data(self) -> Dict[str, Any]:
        return self.summary_metrics_data

    def set_summary_metrics_data(self, data: Dict[str, Any]):
        self.summary_metrics_data = data

    def get_account_metrics_data(self) -> Dict[str, Any]:
        return self.account_metrics_data

    def set_account_metrics_data(self, data: Dict[str, Any]):
        self.account_metrics_data = data

    def get_index_quote_data(self) -> Dict[str, Dict[str, Any]]:
        return self.index_quote_data

    def set_index_quote_data(self, data: Dict[str, Dict[str, Any]]):
        self.index_quote_data = data

    def get_historical_data(self) -> pd.DataFrame:
        return self.historical_data

    def set_historical_data(self, df: pd.DataFrame):
        self.historical_data = df

    def get_ignored_data(self) -> pd.DataFrame:
        return self.ignored_data

    def set_ignored_data(self, df: pd.DataFrame):
        self.ignored_data = df

    def get_dividend_history_data(self) -> pd.DataFrame:
        return self.dividend_history_data

    def set_dividend_history_data(self, df: pd.DataFrame):
        self.dividend_history_data = df

    def get_capital_gains_history_data(self) -> pd.DataFrame:
        return self.capital_gains_history_data

    def set_capital_gains_history_data(self, df: pd.DataFrame):
        self.capital_gains_history_data = df

    def get_correlation_matrix_df(self) -> pd.DataFrame:
        return self.correlation_matrix_df

    def set_correlation_matrix_df(self, df: pd.DataFrame):
        self.correlation_matrix_df = df

    def get_factor_analysis_results(self) -> Optional[Dict]:
        return self.factor_analysis_results

    def set_factor_analysis_results(self, results: Optional[Dict]):
        self.factor_analysis_results = results

    def get_scenario_analysis_result(self) -> Dict:
        return self.scenario_analysis_result

    def set_scenario_analysis_result(self, result: Dict):
        self.scenario_analysis_result = result

    def get_full_historical_data(self) -> pd.DataFrame:
        return self.full_historical_data

    def set_full_historical_data(self, df: pd.DataFrame):
        self.full_historical_data = df

    def get_historical_prices_yf_adjusted(self) -> Dict[str, pd.DataFrame]:
        return self.historical_prices_yf_adjusted

    def set_historical_prices_yf_adjusted(self, data: Dict[str, pd.DataFrame]):
        self.historical_prices_yf_adjusted = data

    def get_historical_fx_yf(self) -> Dict[str, pd.DataFrame]:
        return self.historical_fx_yf

    def set_historical_fx_yf(self, data: Dict[str, pd.DataFrame]):
        self.historical_fx_yf = data

    def get_periodic_returns_data(self) -> Dict[str, pd.DataFrame]:
        return self.periodic_returns_data

    def set_periodic_returns_data(self, data: Dict[str, pd.DataFrame]):
        self.periodic_returns_data = data

    def get_periodic_value_changes_data(self) -> Dict[str, pd.DataFrame]:
        return self.periodic_value_changes_data

    def set_periodic_value_changes_data(self, data: Dict[str, pd.DataFrame]):
        self.periodic_value_changes_data = data
