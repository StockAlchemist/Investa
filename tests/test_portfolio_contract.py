import json
from pathlib import Path

from server.response_models import SummaryResponse, HoldingResponse, HistoryPoint


def test_shared_contract_preserves_dynamic_fields_and_nullable_metrics():
    fixture = json.loads((Path(__file__).parent / "fixtures/portfolio_contract.json").read_text())
    summary = SummaryResponse.model_validate(fixture["summary"])
    assert summary.metrics.annualized_twr is None
    assert summary.freshness.refreshing is True
    holding = HoldingResponse.model_validate(fixture["holdings"][0])
    assert holding.model_dump()["Market Value (USD)"] == 1000
    point = HistoryPoint.model_validate(fixture["history"][0])
    assert point.model_dump()["S&P 500"] == 10
