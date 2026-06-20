import logging
import json
import os
import hashlib
from datetime import datetime
import config
from server.ai_analyzer import FALLBACK_MODELS
import requests

# --- Helpers ---


def _compute_portfolio_hash(portfolio_data: dict, db_conn=None) -> str:
    """Computes a stable hash of the portfolio state for caching.

    When ``db_conn`` is supplied it MUST be the same user-scoped portfolio DB
    that ``generate_portfolio_review`` will read screener rows from — otherwise
    the hash and the review would be fingerprinting different databases and the
    staleness check is meaningless.
    """
    key_components = []

    metrics = portfolio_data.get("metrics", {})
    key_components.append(str(int(metrics.get("market_value", 0))))

    holdings_dict = portfolio_data.get("holdings_dict", {})
    sorted_holdings = sorted(holdings_dict.items())
    symbols: set = set()
    for k, v in sorted_holdings:
        qty = v.get("qty", 0)
        if abs(qty) > 0.01:
            key_components.append(f"{k[0]}:{k[1]}:{qty:.2f}")
            if k and k[0]:
                symbols.add(str(k[0]).upper())

    key_components.append(datetime.now().strftime("%Y-%m-%d"))

    # Fingerprint the screener-cache rows for held symbols so a mid-day IV/MoS
    # refresh busts this AI-review cache. Without this, a cached review that
    # used stale screener data keeps being served until the date rolls over.
    if symbols:
        try:
            from db_utils import get_cached_screener_results

            rows = get_cached_screener_results(list(symbols)) or {}
            stamps = [
                str(row.get("updated_at"))
                for row in rows.values()
                if row and row.get("updated_at")
            ]
            if stamps:
                key_components.append("scr:" + max(stamps))
        except Exception as e:
            logging.warning(
                f"Portfolio AI hash: could not fingerprint screener cache: {e}"
            )

    combined_str = "|".join(key_components)
    return hashlib.md5(combined_str.encode()).hexdigest()


def _detect_tax_loss_candidates(holdings: list) -> list:
    """Identifies positions with significant unrealized losses for TLH suggestions."""
    candidates = []
    for h in holdings:
        gain = h.get("unrealized_gain", 0)
        symbol = h.get("symbol")
        if gain < -500 or (
            h.get("market_value", 0) > 0
            and gain / (h.get("market_value", 0) - gain) < -0.10
        ):
            candidates.append(
                {
                    "symbol": symbol,
                    "loss": gain,
                    "loss_percent": (
                        (gain / (h.get("market_value", 0) - gain) * 100)
                        if (h.get("market_value", 0) - gain) != 0
                        else 0
                    ),
                }
            )
    return sorted(candidates, key=lambda x: x["loss"])


def _calculate_sector_allocation(holdings: list) -> dict:
    """Groups holdings by sector to identify over-concentration."""
    sectors = {}
    total_val = sum(h.get("market_value", 0) for h in holdings)
    if total_val == 0:
        return {}

    for h in holdings:
        s = h.get("sector", "Unknown")
        val = h.get("market_value", 0)
        sectors[s] = sectors.get(s, 0) + val

    return {s: (v / total_val * 100) for s, v in sectors.items()}


def generate_portfolio_review(
    portfolio_data: dict,
    risk_metrics: dict,
    force_refresh: bool = False,
    db_conn=None,
) -> dict:
    """
    Generates an AI review for the entire portfolio.

    Args:
        portfolio_data (dict): The result from calculate_portfolio_summary
        risk_metrics (dict): The result from risk_metrics.calculate_all_risk_metrics
        force_refresh (bool): Whether to bypass cache
        db_conn: The user-scoped portfolio DB connection used to look up
            intrinsic-value / margin-of-safety rows from ``screener_cache``.
            MUST be the user's DB — the default ``get_db_connection()`` returns
            the shared ``data/db/portfolio.db`` which can hold days-old rows
            and produces an MoS the user will see nowhere else in the UI.

    Returns:
        dict: The AI review JSON (scorecard + analysis)
    """
    logging.info("AI Analysis: Generating PORTFOLIO review...")

    # 1. Check Cache
    cache_dir = os.path.join(
        config.get_app_data_dir(), config.CACHE_DIR, "portfolio_ai_cache"
    )
    os.makedirs(cache_dir, exist_ok=True)

    pf_hash = _compute_portfolio_hash(portfolio_data, db_conn=db_conn)
    cache_path = os.path.join(cache_dir, f"pf_{pf_hash}.json")

    if not force_refresh and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
                # Check TTL (e.g. 24 hours) - though hash includes date so it auto-expires daily
                logging.info("AI Analysis: Using cached portfolio review.")
                return cached
        except Exception as e:
            logging.warning(f"Failed to read portfolio cache: {e}")

    # 2. Prepare Data for Prompt
    metrics = portfolio_data.get("metrics", {})

    # Top Holdings Calculation
    _holdings_raw = portfolio_data.get("holdings_dict", {})

    # Removed unused holdings iteration that did not compute aggregated positions.

    # We will pass the TOP 10 positions by weight if possible.
    # Since we might not have prices easily here without reprocessing,
    # we can rely on what's available or ask the caller to pass a simplified view.
    # For now, let's assume we can dump the 'metrics' and 'risk_metrics' which are high level.

    # 2. Extract key metrics
    # Map from API portfolio summary keys to analyzer keys
    metrics = portfolio_data.get("metrics", {})
    total_value = metrics.get("market_value", 0)
    total_change = metrics.get("total_gain", 0)
    total_change_percent = metrics.get("total_return_pct", 0)

    # Risk Metrics - Map keys from risk_metrics.py
    # Keys seen in log: 'Max Drawdown', 'Volatility (Ann.)', 'Sharpe Ratio', 'Sortino Ratio'
    sharpe = risk_metrics.get("Sharpe Ratio", risk_metrics.get("sharpe_ratio", "N/A"))
    sortino = risk_metrics.get(
        "Sortino Ratio", risk_metrics.get("sortino_ratio", "N/A")
    )
    volatility = risk_metrics.get(
        "Volatility (Ann.)", risk_metrics.get("volatility", "N/A")
    )
    max_drawdown = risk_metrics.get(
        "Max Drawdown", risk_metrics.get("max_drawdown", "N/A")
    )
    _beta = risk_metrics.get(
        "Beta", risk_metrics.get("beta", "N/A")
    )  # Beta might be missing in pure portfolio stats
    alpha = risk_metrics.get("Alpha", risk_metrics.get("alpha", "N/A"))

    # Asset Allocation (if available)
    holdings = portfolio_data.get("holdings", [])
    # Group by sector if available, otherwise just list top holdings

    holdings_summary = ""
    sorted_holdings: list = []
    if holdings:
        sorted_holdings = sorted(
            holdings,
            key=lambda x: x.get("market_value", x.get("value", 0)) or 0,
            reverse=True,
        )

        # Enrich with intrinsic-value + moat signals from the screener cache
        # so the prompt can ground "value discipline" judgements in concrete data
        # rather than guessing from price alone. Reads go through the global
        # screener DB so every user sees the same canonical IV/MoS row.
        iv_lookup: dict = {}
        try:
            from db_utils import get_cached_screener_results

            top_symbols = [
                h.get("symbol") for h in sorted_holdings[:10] if h.get("symbol")
            ]
            if top_symbols:
                iv_lookup = get_cached_screener_results(top_symbols) or {}
        except Exception as e_iv:
            logging.warning(
                f"Portfolio AI: could not enrich with screener-cache signals: {e_iv}"
            )

        holdings_summary = "Top 10 Holdings (weight, sector, intrinsic-value signals where available):\n"
        for h in sorted_holdings[:10]:
            symbol = h.get("symbol", "?")
            val = h.get("market_value", h.get("value", 0)) or 0
            pct = h.get("allocation_percent", h.get("percent", 0)) or 0
            if pct == 0 and total_value > 0:
                pct = (val / total_value) * 100
            sector = h.get("sector", "Unknown")

            iv_data = iv_lookup.get(str(symbol).upper(), {}) if symbol else {}
            iv = iv_data.get("intrinsic_value")
            mos = iv_data.get("margin_of_safety")
            moat = iv_data.get("ai_moat")

            iv_bits = []
            if iv is not None:
                try:
                    iv_bits.append(f"IV {float(iv):,.2f}")
                except (TypeError, ValueError):
                    pass
            if mos is not None:
                try:
                    iv_bits.append(f"MoS {float(mos) * 1.0:+.0f}%")
                except (TypeError, ValueError):
                    pass
            if moat is not None:
                try:
                    iv_bits.append(f"Moat {float(moat):.1f}/10")
                except (TypeError, ValueError):
                    pass
            iv_str = f" — {', '.join(iv_bits)}" if iv_bits else ""

            holdings_summary += f"- {symbol}: {pct:.2f}% ({sector}){iv_str}\n"
    else:
        holdings_summary = "Holdings data not explicit."

    # 3. Optimization Data
    tlh_candidates = _detect_tax_loss_candidates(holdings)
    sector_alloc = _calculate_sector_allocation(holdings)

    tlh_summary = (
        "\n".join(
            [
                f"- {c['symbol']}: {c['loss']:,.2f} ({c['loss_percent']:.1f}%)"
                for c in tlh_candidates
            ]
        )
        if tlh_candidates
        else "No major tax-loss candidates found."
    )
    sector_summary = "\n".join(
        [
            f"- {s}: {v:.1f}%"
            for s, v in sorted(sector_alloc.items(), key=lambda x: x[1], reverse=True)
        ]
    )

    # 4. Construct Prompt
    prompt = f"""
You are an investment advisor working in the quality-and-value tradition of Buffett, Munger, Phil Fisher, Terry Smith, and Nick Sleep.
The investor you serve runs a deliberately concentrated portfolio of high-conviction businesses. Treat this as the "punch card" approach to investing — fewer, bigger, longer-held positions in businesses the investor genuinely understands.

PHILOSOPHICAL GROUND RULES (these are non-negotiable):

  1. Concentration is conviction, not error. A 20–40% position in a great business is correct. Diversification is, in Buffett's words, "protection against ignorance" — do NOT recommend it for its own sake, and never score the portfolio down for being concentrated.
  2. Volatility is not risk. Real risk is the permanent impairment of capital, not price fluctuation. Beta, annualised volatility, and max-drawdown do NOT inform business quality. They are reference numbers only.
  3. Sector weight reflects circle of competence. Heavy exposure to a few sectors is by design. Do not flag it.
  4. The only valid reasons to sell are: (a) the business has fundamentally deteriorated, (b) price has run materially above any reasonable estimate of intrinsic value, or (c) a clearly superior opportunity exists and capital must be redeployed. "Rebalancing", "hedging", and "tax-loss harvesting" are NOT, on their own, valid reasons to sell a great business.
  5. Inactivity is a virtue. The default recommendation, in the absence of a strong reason, is HOLD.
  6. Where Margin of Safety (MoS) data exists in the data below, treat NEGATIVE MoS (price ABOVE intrinsic value) as the primary signal that a position may warrant trimming, and treat DEEPLY POSITIVE MoS as the primary signal a high-quality holding may warrant adding to.

Anti-patterns to avoid in your output:
  • "You should diversify into bonds / other sectors / international equity to reduce risk."
  • "Position X is too large at Y% — trim to reduce concentration."
  • "Portfolio beta is high — consider a hedge."
  • "Add an S&P 500 ETF for broad market exposure."
  • Generic advisor boilerplate about "balanced portfolios" or "age-appropriate allocation".
If you find yourself writing any of the above, delete it and rewrite.

—————————————————————————————————————————————————
PORTFOLIO SNAPSHOT
—————————————————————————————————————————————————
Total Value: {total_value:,.2f}
Cumulative Return: {total_change:,.2f} ({total_change_percent:.2f}%)
Alpha (vs market): {alpha}
Sharpe / Sortino (reference only): {sharpe} / {sortino}
Annualised Vol / Max DD (reference only, not a quality signal): {volatility} / {max_drawdown}

{holdings_summary}
Sector weights (informational — concentration is fine):
{sector_summary}

POSITIONS CARRYING UNREALISED LOSSES:
{tlh_summary}
(For each: ask whether the business has actually deteriorated, or whether Mr. Market is offering a discount on a name still worth owning. A losing price is not, by itself, a reason to sell.)

—————————————————————————————————————————————————
WHAT TO PRODUCE
—————————————————————————————————————————————————

SCORECARD (integers 1–10):

  • business_quality — Are these businesses with durable moats, high returns on incremental invested capital, and predictable owner-earnings? Cite specific tickers that drive the score up or down.
  • value_discipline — Are positions trading at or below reasonable intrinsic value? Weight Margin-of-Safety data heavily where present. A holding trading 40% above IV pulls this score down regardless of how great the business is.
  • thesis_integrity — Do the holdings hang together as a coherent set of identifiable theses (e.g. "compounders with pricing power", "regional banks at tangible book", "Asian consumer franchise"), or is the portfolio a diffuse, theme-less collection?

EXECUTIVE SUMMARY (3–5 sentences):
Speak about businesses owned, not statistical aggregates. The reader is the investor who chose every name; address them as a peer.

PER-DIMENSION ANALYSIS:
For each scorecard dimension, write 3–6 sentences. Be specific. Cite tickers. Distinguish what is strengthening the score from what is dragging on it, on FUNDAMENTAL grounds.

OPPORTUNITIES (a list of concrete actions):
Allowed action types:
  • "add"           — a high-quality holding trading meaningfully below IV. Compounding opportunity.
  • "trim"          — a holding now well above IV; risk/reward has flipped. NOT because the position is "too big".
  • "exit"          — the business has fundamentally deteriorated, OR the original thesis was wrong.
  • "monitor"       — early-warning signs (margin pressure, slowing reinvestment, governance, competitive shifts).
  • "tax_efficiency" — ONLY when a position with an unrealised loss is also a candidate for exit on fundamentals. Never harvest losses on a business you still believe in.

For each item: 2–3 sentences of rationale, anchored in the business — not the chart.

—————————————————————————————————————————————————
OUTPUT — strict JSON, no markdown, no commentary outside the JSON:
—————————————————————————————————————————————————
{{
    "scorecard": {{
        "business_quality": <int 1-10>,
        "value_discipline": <int 1-10>,
        "thesis_integrity": <int 1-10>
    }},
    "summary": "<3-5 sentence executive summary>",
    "analysis": {{
        "business_quality": "<3-6 sentences, cite tickers>",
        "value_discipline": "<3-6 sentences, cite tickers, reference MoS where present>",
        "thesis_integrity": "<3-6 sentences>",
        "actionable_recommendations": "<summary of the most important next moves, in plain prose>"
    }},
    "recommendations": ["<short rec 1>", "<short rec 2>", "..."],
    "optimizations": [
        {{
            "type": "add" | "trim" | "exit" | "monitor" | "tax_efficiency",
            "title": "<short_title>",
            "description": "<2-3 sentence rationale rooted in business fundamentals>",
            "symbol": "<ticker>",
            "action": "Add" | "Trim" | "Sell" | "Hold" | "Monitor",
            "priority": "High" | "Medium" | "Low"
        }}
    ]
}}
"""

    # 3. Call LLM (using Fallback Chain from ai_analyzer)
    api_key = config.GEMINI_API_KEY
    if not api_key:
        return {
            "scorecard": {
                "business_quality": 0,
                "value_discipline": 0,
                "thesis_integrity": 0,
            },
            "summary": "Unable to generate analysis. Error: API Key is missing.",
            "analysis": {
                "business_quality": "Analysis unavailable.",
                "value_discipline": "Analysis unavailable.",
                "thesis_integrity": "Analysis unavailable.",
                "actionable_recommendations": "No recommendations available.",
            },
            "recommendations": [],
            "optimizations": [],
        }

    base_payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json"},
    }

    rate_limit_count = 0
    other_error_count = 0

    for model in FALLBACK_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        try:
            resp = requests.post(url, json=base_payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]

                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]

                result = json.loads(text)

                default_scorecard = {
                    "business_quality": 0,
                    "value_discipline": 0,
                    "thesis_integrity": 0,
                }
                default_analysis = {
                    "business_quality": "Analysis unavailable.",
                    "value_discipline": "Analysis unavailable.",
                    "thesis_integrity": "Analysis unavailable.",
                    "actionable_recommendations": "No recommendations available.",
                }

                if "scorecard" not in result:
                    result["scorecard"] = default_scorecard
                if "analysis" not in result:
                    result["analysis"] = default_analysis
                if "summary" not in result:
                    result["summary"] = "AI analysis generated, but summary missing."
                if "recommendations" not in result:
                    result["recommendations"] = []
                if "optimizations" not in result:
                    result["optimizations"] = []

                # Stamp generation time so the UI can show freshness/staleness.
                result["generated_at"] = datetime.now().isoformat()

                # Cache it
                with open(cache_path, "w") as f:
                    json.dump(result, f)

                return result
            elif resp.status_code == 429:
                logging.warning(f"Portfolio AI: Model {model} rate limited.")
                rate_limit_count += 1
                if "exceeded your current quota" in resp.text:
                    logging.warning(f"Hard quota limit reached for {model}. Stopping fallback to prevent charges.")
                    break
                continue
            else:
                other_error_count += 1
                logging.warning(
                    f"Portfolio AI: Model {model} error {resp.status_code}: {resp.text}"
                )

        except Exception as e:
            logging.error(f"Portfolio AI: Error with {model}: {e}")
            other_error_count += 1
            continue

    if rate_limit_count > 0 and other_error_count == 0:
        # Try to fallback to cache if available
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cached = json.load(f)
                    cached["warning"] = "RateLimit"
                    cached["message"] = (
                        "AI service is busy. Showing cached analysis from earlier."
                    )
                    logging.info("AI Analysis: Rate limited, falling back to cache.")
                    return cached
            except Exception as e:
                logging.warning(f"Failed to read portfolio cache for fallback: {e}")

        return {
            "error": "RateLimit",
            "message": "AI service usage limit reached. Please try again in a few minutes or check your quota.",
        }

    return {"error": "Failed to generate portfolio review."}
