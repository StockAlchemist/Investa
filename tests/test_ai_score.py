import os
import sys
import json
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "server")))

from screener_service import process_screener_results

@pytest.fixture
def mock_app_data_dir():
    temp_dir = tempfile.mkdtemp()
    cache_dir = os.path.join(temp_dir, "cache", "ai_analysis_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    with patch("config.get_app_data_dir", return_value=temp_dir):
        # We need to also patch the constant CACHE_DIR if config uses it directly,
        # but in process_screener_results it uses: 
        # os.path.join(config.get_app_data_dir(), config.CACHE_DIR, "ai_analysis_cache")
        # Let's mock config.CACHE_DIR to be "cache"
        with patch("config.CACHE_DIR", "cache"):
            yield cache_dir
            
    shutil.rmtree(temp_dir)

def test_process_screener_results_ai_score_basic(mock_app_data_dir):
    sym = "TESTSYM"
    # Create mock JSON in the temp cache dir
    ai_json = {
        "analysis": {
            "scorecard": {
                "moat": 8.0,
                "financial_strength": 7.0,
                "predictability": 9.0,
                "growth": 6.0
            },
            "summary": "This is a good stock."
        }
    }
    with open(os.path.join(mock_app_data_dir, f"{sym}_analysis.json"), "w") as f:
        json.dump(ai_json, f)
        
    symbols = [sym]
    quotes = {sym: {"price": 100.0}}
    details = {sym: {"lastFiscalYearEnd": 123456, "mostRecentQuarter": 123456}}
    
    # Needs db_conn=None, prefetched_statements={}
    # But wait, it calls get_shared_mdp() -> we can mock that if it tries to recalculate.
    # If cached_map is passed without matching IV, it tries to recalculate. 
    # Let's pass a matched cache to avoid hitting get_shared_mdp() to recalculate intrinsic value.
    cached_map = {
        sym: {
            "last_fiscal_year_end": 123456,
            "most_recent_quarter": 123456,
            "intrinsic_value": 120.0
        }
    }
    
    results = process_screener_results(symbols, quotes, details, cached_map=cached_map)
    
    assert len(results) == 1
    res = results[0]
    
    assert res["symbol"] == sym
    assert res["has_ai_review"] == True
    # (8 + 7 + 9 + 6) / 4 = 7.5
    assert res["ai_score"] == 7.5
    assert res["ai_moat"] == 8.0
    assert res["ai_financial_strength"] == 7.0
    assert res["ai_predictability"] == 9.0
    assert res["ai_growth"] == 6.0
    assert res["ai_summary"] == "This is a good stock."

def test_process_screener_results_ai_score_missing_fields(mock_app_data_dir):
    sym = "TESTSYM2"
    # Create mock JSON in the temp cache dir with missing numeric fields or non-numeric
    ai_json = {
        "analysis": {
            "scorecard": {
                "moat": 8.0,
                "financial_strength": None,
                "predictability": "High",  # Should be omitted from aggregation
                "growth": 4.0
            }
        }
    }
    with open(os.path.join(mock_app_data_dir, f"{sym}_analysis.json"), "w") as f:
        json.dump(ai_json, f)
        
    symbols = [sym]
    quotes = {sym: {"price": 100.0}}
    details = {sym: {"lastFiscalYearEnd": 123456, "mostRecentQuarter": 123456}}
    cached_map = {sym: {"last_fiscal_year_end": 123456, "most_recent_quarter": 123456, "intrinsic_value": 120.0}}
    
    results = process_screener_results(symbols, quotes, details, cached_map=cached_map)
    
    assert len(results) == 1
    res = results[0]
    
    # (8.0 + 4.0) / 2 = 6.0
    assert res["ai_score"] == 6.0
    assert res["ai_moat"] == 8.0
    assert res["ai_financial_strength"] == None
    assert res["ai_predictability"] == "High"
    assert res["ai_growth"] == 4.0
