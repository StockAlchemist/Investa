import Foundation

/// `GET /api/projection` — forward portfolio-value projection (lognormal model).
struct Projection: Codable, Sendable {
    let available: Bool
    let currentValue: Double?
    /// Geometric (median) annualized return used for the projection, in percent.
    let annualReturnPct: Double?
    let annualVolatilityPct: Double?
    let currency: String?
    let horizons: [ProjectionHorizon]?

    enum CodingKeys: String, CodingKey {
        case available
        case currentValue = "current_value"
        case annualReturnPct = "annual_return_pct"
        case annualVolatilityPct = "annual_volatility_pct"
        case currency
        case horizons
    }
}

/// One horizon's projected value: the median plus 10/25/75/90th percentile bands.
struct ProjectionHorizon: Codable, Sendable, Identifiable {
    let years: Int
    let medianValue: Double
    let medianReturnPct: Double
    let expectedValue: Double
    let p10: Double
    let p25: Double
    let p75: Double
    let p90: Double

    var id: Int { years }

    enum CodingKeys: String, CodingKey {
        case years
        case medianValue = "median_value"
        case medianReturnPct = "median_return_pct"
        case expectedValue = "expected_value"
        case p10, p25, p75, p90
    }
}

/// `GET /api/projection/backtest` — walk-forward backtest of the projection
/// model on the portfolio's own history: refit at each past month on the data
/// that existed then, then scored against what actually followed.
struct ProjectionBacktest: Codable, Sendable {
    let available: Bool
    /// Why it is unavailable (`insufficient_history`, `no_history`, …).
    let reason: String?
    let currency: String?
    /// The zone the dates below were reckoned in (the market clock, not the device).
    let marketTimezone: String?
    let historyYears: Double?
    let historyStart: String?
    let historyEnd: String?
    /// History a backtest needs: the fitting window plus the shortest horizon.
    let requiredYears: Double?
    let horizons: [ProjectionBacktestHorizon]?
    let replay: ProjectionReplay?

    enum CodingKeys: String, CodingKey {
        case available, reason, currency, horizons, replay
        case marketTimezone = "market_timezone"
        case historyYears = "history_years"
        case historyStart = "history_start"
        case historyEnd = "history_end"
        case requiredYears = "required_years"
    }
}

/// Calibration of one horizon: how often the outcome actually landed inside the
/// bands the model drew. `inBandPct` should be near 80.
struct ProjectionBacktestHorizon: Codable, Sendable, Identifiable {
    let years: Int
    let samples: Int
    /// Spread of the standardized errors; 1.0 means the bands were exactly right.
    let stdZ: Double
    let inBandPct: Double
    let belowP10Pct: Double
    let aboveP90Pct: Double
    let meanU: Double
    let medianActualReturnPct: Double
    let medianProjectedReturnPct: Double
    /// `calibrated`, `narrow` (overconfident) or `wide` (conservative).
    let verdict: String

    var id: Int { years }

    enum CodingKeys: String, CodingKey {
        case years, samples, verdict
        case stdZ = "std_z"
        case inBandPct = "in_band_pct"
        case belowP10Pct = "below_p10_pct"
        case aboveP90Pct = "above_p90_pct"
        case meanU = "mean_u"
        case medianActualReturnPct = "median_actual_return_pct"
        case medianProjectedReturnPct = "median_projected_return_pct"
    }
}

/// The cone the model drew `years` ago, with the path actually taken since.
struct ProjectionReplay: Codable, Sendable {
    let anchorDate: String
    let years: Double
    let startValue: Double
    /// True when the portfolio's value then was unknown, so the replay starts at 100.
    let indexed: Bool
    let fitYears: Double
    let annualReturnPct: Double
    let annualVolatilityPct: Double
    let finalActual: Double
    let finalMedian: Double
    let finalP10: Double
    let finalP90: Double
    /// Where the realized outcome landed: `inside`, `below` or `above` the band.
    let outcome: String
    let points: [ProjectionReplayPoint]

    enum CodingKeys: String, CodingKey {
        case years, indexed, outcome, points
        case anchorDate = "anchor_date"
        case startValue = "start_value"
        case fitYears = "fit_years"
        case annualReturnPct = "annual_return_pct"
        case annualVolatilityPct = "annual_volatility_pct"
        case finalActual = "final_actual"
        case finalMedian = "final_median"
        case finalP10 = "final_p10"
        case finalP90 = "final_p90"
    }
}

/// One month of a replay: the bands projected for that date, and what happened.
struct ProjectionReplayPoint: Codable, Sendable, Identifiable {
    let date: String
    let years: Double
    /// Nil only if the history has a hole where this month should be.
    let actual: Double?
    let median: Double
    let p10: Double
    let p25: Double
    let p75: Double
    let p90: Double

    var id: String { date }

    /// The point's calendar day, parsed on the market clock (never the device's).
    var day: Date? { MarketTime.calendarDay(date) }

    enum CodingKeys: String, CodingKey {
        case date, years, actual, median, p10, p25, p75, p90
    }
}
