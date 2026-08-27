# -*- coding: utf-8 -*-
"""Yahoo-style FX pair names, and crossing a quoted table into them.

The archive stores rates under Yahoo's naming, which `portfolio_history` and
`finutils` already depend on meaning a particular thing:

    THB=X      THB per USD     (three letters: USD is the implied base)
    USDTHB=X   THB per USD     (the same series, spelled out)
    THBUSD=X   USD per THB     (the inverse)

Official feeds do not publish in that shape. Each quotes every currency against
its own unit — the ECB against the euro, the Bank of Thailand against the baht —
so a pair is a cross through whatever that unit happens to be, and the only
thing that differs between providers is which currency the table is quoted in.
That is why this lives on its own rather than inside either provider.
"""

from typing import Dict, Iterable, List, Optional, Tuple

# {'yyyy-MM-dd': {'USD': 1.1662, 'THB': 38.176, ...}}, each value the number of
# that currency per one unit of the table's own currency.
RateTable = Dict[str, Dict[str, float]]


def split_pair(pair: str) -> Optional[Tuple[str, str]]:
    """Pair name -> (base, quote), or None if it is not one.

    A three-letter name is read as `USD{CUR}`, which is what makes `USD=X`
    resolve to a flat 1.0 rather than to whatever a provider returns for a
    currency against itself.
    """
    if not pair or not pair.upper().endswith("=X"):
        return None
    code = pair.upper()[:-2]
    if len(code) == 3 and code.isalpha():
        return "USD", code
    if len(code) == 6 and code.isalpha():
        return code[:3], code[3:]
    return None


def pair_rate(row: Dict[str, float], pair: str, unit: str) -> Optional[float]:
    """The pair's rate for one day's row, or None if the day cannot price it.

    `unit` is the currency the row is quoted in; it never appears as a key in
    the row, because a table does not quote its own unit against itself.

    Both legs cross through that unit, so a pair is available only on a day the
    provider published *both* currencies — which is why the ECB can price
    EUR/JPY back to 1999 but THB only from Apr 2005.
    """
    legs = split_pair(pair)
    if not legs:
        return None
    base, quote = legs

    def leg(code: str) -> Optional[float]:
        if code == unit:
            return 1.0
        value = row.get(code)
        return value if value and value > 0 else None

    base_rate, quote_rate = leg(base), leg(quote)
    if base_rate is None or quote_rate is None:
        return None
    return quote_rate / base_rate


def pair_series(rates: RateTable, pair: str, unit: str) -> List[Tuple[str, float]]:
    """(date, rate) for every day the pair is derivable, oldest first."""
    out = [
        (day, value)
        for day, row in rates.items()
        if (value := pair_rate(row, pair, unit)) is not None
    ]
    out.sort()
    return out


def supported_pairs(
    rates: RateTable, candidates: Iterable[str], unit: str
) -> List[str]:
    """Those candidates the feed can actually price on at least one day."""
    return [
        p for p in candidates if any(pair_rate(row, p, unit) for row in rates.values())
    ]
