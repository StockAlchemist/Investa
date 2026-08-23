import Foundation

struct StockParamConfig: Identifiable, Sendable {
    var id: String { key }
    let key: String
    let label: String
    let description: String
    let unit: ParamUnit
    let step: Double
    let min: Double
    let max: Double
    let isPercent: Bool

    enum ParamUnit: Sendable {
        case percent
        case currency
        case multiple
        case years
        case number
    }
}

struct BlendedValuationResult: Sendable {
    let customAverage: Double?
    let customMarginOfSafety: Double?
    let customModelValues: [String: Double]
    let hasAnyCustom: Bool
}

enum StockValuationCalculator {

    static let configs: [String: [StockParamConfig]] = [
        "dcf": [
            StockParamConfig(
                key: "discount_rate",
                label: "Discount Rate (WACC)",
                description: "Discount rate to convert future cash flows to present value.",
                unit: .percent,
                step: 0.0025,
                min: 0.01,
                max: 0.30,
                isPercent: true
            ),
            StockParamConfig(
                key: "growth_rate",
                label: "Growth Rate",
                description: "Expected explicit annual growth of free cash flows.",
                unit: .percent,
                step: 0.005,
                min: -0.20,
                max: 0.50,
                isPercent: true
            ),
            StockParamConfig(
                key: "terminal_growth_rate",
                label: "Terminal Growth Rate",
                description: "Long-term perpetual growth rate in mature stage.",
                unit: .percent,
                step: 0.0025,
                min: 0.0,
                max: 0.08,
                isPercent: true
            ),
            StockParamConfig(
                key: "projection_years",
                label: "Projection Years",
                description: "Number of explicit projection years.",
                unit: .years,
                step: 1.0,
                min: 3.0,
                max: 25.0,
                isPercent: false
            ),
            StockParamConfig(
                key: "base_fcf",
                label: "Base Free Cash Flow",
                description: "Starting annual Free Cash Flow value in native currency.",
                unit: .currency,
                step: 1_000_000.0,
                min: 1.0,
                max: 1_000_000_000_000.0,
                isPercent: false
            )
        ],
        "dcfo": [
            StockParamConfig(
                key: "cfo_per_share",
                label: "Base CFO / Share",
                description: "Starting Operating Cash Flow per share.",
                unit: .currency,
                step: 0.1,
                min: 0.01,
                max: 10000.0,
                isPercent: false
            ),
            StockParamConfig(
                key: "growth_rate",
                label: "CFO Growth Rate",
                description: "Expected growth rate for Operating Cash Flow.",
                unit: .percent,
                step: 0.005,
                min: -0.20,
                max: 0.40,
                isPercent: true
            ),
            StockParamConfig(
                key: "discount_rate",
                label: "Discount Rate (WACC)",
                description: "Cost of capital discount rate.",
                unit: .percent,
                step: 0.0025,
                min: 0.02,
                max: 0.30,
                isPercent: true
            ),
            StockParamConfig(
                key: "terminal_growth_rate",
                label: "Terminal Growth Rate",
                description: "Long-term sustainable CFO growth rate.",
                unit: .percent,
                step: 0.0025,
                min: 0.0,
                max: 0.08,
                isPercent: true
            )
        ],
        "dni": [
            StockParamConfig(
                key: "base_eps",
                label: "Base EPS",
                description: "Starting normalized Earnings Per Share.",
                unit: .currency,
                step: 0.1,
                min: 0.01,
                max: 10000.0,
                isPercent: false
            ),
            StockParamConfig(
                key: "growth_rate",
                label: "Net Income Growth",
                description: "Projected annual Net Income / EPS growth rate.",
                unit: .percent,
                step: 0.005,
                min: -0.20,
                max: 0.40,
                isPercent: true
            ),
            StockParamConfig(
                key: "discount_rate",
                label: "Cost of Equity",
                description: "Discount rate (Cost of Equity via CAPM).",
                unit: .percent,
                step: 0.0025,
                min: 0.03,
                max: 0.30,
                isPercent: true
            ),
            StockParamConfig(
                key: "terminal_growth_rate",
                label: "Terminal Growth Rate",
                description: "Perpetual sustainable earnings growth rate.",
                unit: .percent,
                step: 0.0025,
                min: 0.0,
                max: 0.08,
                isPercent: true
            )
        ],
        "ddm": [
            StockParamConfig(
                key: "base_dividend",
                label: "Base Annual Dividend",
                description: "Starting annual dividend per share.",
                unit: .currency,
                step: 0.05,
                min: 0.01,
                max: 1000.0,
                isPercent: false
            ),
            StockParamConfig(
                key: "growth_rate",
                label: "Dividend Growth Rate",
                description: "Projected annual dividend growth rate.",
                unit: .percent,
                step: 0.0025,
                min: -0.10,
                max: 0.25,
                isPercent: true
            ),
            StockParamConfig(
                key: "discount_rate",
                label: "Cost of Equity (CAPM)",
                description: "Required rate of return on equity capital.",
                unit: .percent,
                step: 0.0025,
                min: 0.03,
                max: 0.25,
                isPercent: true
            ),
            StockParamConfig(
                key: "terminal_growth_rate",
                label: "Terminal Growth Rate",
                description: "Long-term perpetual dividend growth rate.",
                unit: .percent,
                step: 0.0025,
                min: 0.0,
                max: 0.06,
                isPercent: true
            )
        ],
        "mean_pe": [
            StockParamConfig(
                key: "eps",
                label: "TTM EPS",
                description: "Trailing twelve months Earnings Per Share.",
                unit: .currency,
                step: 0.1,
                min: 0.01,
                max: 10000.0,
                isPercent: false
            ),
            StockParamConfig(
                key: "applied_pe",
                label: "Applied P/E Multiple",
                description: "Target historical or peer P/E multiple.",
                unit: .multiple,
                step: 0.5,
                min: 5.0,
                max: 60.0,
                isPercent: false
            )
        ],
        "peg": [
            StockParamConfig(
                key: "eps",
                label: "TTM EPS",
                description: "Trailing twelve months Earnings Per Share.",
                unit: .currency,
                step: 0.1,
                min: 0.01,
                max: 10000.0,
                isPercent: false
            ),
            StockParamConfig(
                key: "growth_rate_pct",
                label: "Earnings Growth Rate (%)",
                description: "Expected earnings growth rate (e.g. 15 for 15%).",
                unit: .percent,
                step: 0.5,
                min: 3.0,
                max: 50.0,
                isPercent: false
            ),
            StockParamConfig(
                key: "target_peg",
                label: "Target PEG Ratio",
                description: "Target Price/Earnings-to-Growth ratio (default 1.0x).",
                unit: .multiple,
                step: 0.1,
                min: 0.4,
                max: 3.0,
                isPercent: false
            )
        ],
        "mean_pb": [
            StockParamConfig(
                key: "book_value_per_share",
                label: "Book Value / Share",
                description: "Net Asset / Book Value per share.",
                unit: .currency,
                step: 0.1,
                min: 0.01,
                max: 10000.0,
                isPercent: false
            ),
            StockParamConfig(
                key: "applied_pb",
                label: "Applied P/B Target",
                description: "Target Price-to-Book benchmark multiple.",
                unit: .multiple,
                step: 0.05,
                min: 0.4,
                max: 10.0,
                isPercent: false
            )
        ],
        "mean_ps": [
            StockParamConfig(
                key: "sales_per_share",
                label: "Sales / Share",
                description: "Revenue per share.",
                unit: .currency,
                step: 0.1,
                min: 0.01,
                max: 10000.0,
                isPercent: false
            ),
            StockParamConfig(
                key: "applied_ps",
                label: "Applied P/S Multiple",
                description: "Target Price-to-Sales multiple.",
                unit: .multiple,
                step: 0.1,
                min: 0.4,
                max: 20.0,
                isPercent: false
            )
        ],
        "psg": [
            StockParamConfig(
                key: "sales_per_share",
                label: "Sales / Share",
                description: "Revenue per share.",
                unit: .currency,
                step: 0.1,
                min: 0.01,
                max: 10000.0,
                isPercent: false
            ),
            StockParamConfig(
                key: "revenue_growth_pct",
                label: "Revenue Growth (%)",
                description: "Top-line revenue growth rate (%).",
                unit: .percent,
                step: 0.5,
                min: 3.0,
                max: 60.0,
                isPercent: false
            ),
            StockParamConfig(
                key: "gross_margin_pct",
                label: "Gross Margin (%)",
                description: "Gross profit margin (%).",
                unit: .percent,
                step: 0.5,
                min: 10.0,
                max: 99.0,
                isPercent: false
            ),
            StockParamConfig(
                key: "target_psg",
                label: "Target PSG Ratio",
                description: "Target Price-to-Sales Growth ratio (default 1.0x).",
                unit: .multiple,
                step: 0.1,
                min: 0.4,
                max: 3.0,
                isPercent: false
            )
        ],
        "graham": [
            StockParamConfig(
                key: "eps",
                label: "Trailing EPS",
                description: "Earnings per share.",
                unit: .currency,
                step: 0.1,
                min: 0.01,
                max: 10000.0,
                isPercent: false
            ),
            StockParamConfig(
                key: "growth_rate_pct",
                label: "Expected Growth (g)",
                description: "7-10 year expected annual growth rate (%).",
                unit: .percent,
                step: 0.5,
                min: -5.0,
                max: 30.0,
                isPercent: false
            ),
            StockParamConfig(
                key: "bond_yield_proxy",
                label: "AAA Bond Yield (Y)",
                description: "Current yield on AAA corporate bonds / 10Y Treasury (%).",
                unit: .percent,
                step: 0.1,
                min: 2.0,
                max: 15.0,
                isPercent: false
            )
        ],
        "lynch": [
            StockParamConfig(
                key: "eps",
                label: "Trailing EPS",
                description: "Earnings per share.",
                unit: .currency,
                step: 0.1,
                min: 0.01,
                max: 10000.0,
                isPercent: false
            ),
            StockParamConfig(
                key: "growth_rate_pct",
                label: "Earnings Growth Rate (%)",
                description: "Long-term earnings growth rate (%).",
                unit: .percent,
                step: 0.5,
                min: 1.0,
                max: 50.0,
                isPercent: false
            ),
            StockParamConfig(
                key: "dividend_yield_pct",
                label: "Dividend Yield (%)",
                description: "Dividend yield rate (%).",
                unit: .percent,
                step: 0.1,
                min: 0.0,
                max: 20.0,
                isPercent: false
            )
        ],
        "epv": [
            StockParamConfig(
                key: "normalized_ebit",
                label: "Normalized Operating EBIT",
                description: "Normalized steady-state operating income (EBIT).",
                unit: .currency,
                step: 1_000_000.0,
                min: 1.0,
                max: 1_000_000_000_000.0,
                isPercent: false
            ),
            StockParamConfig(
                key: "discount_rate",
                label: "Cost of Capital",
                description: "Weighted average cost of capital (WACC).",
                unit: .percent,
                step: 0.0025,
                min: 0.03,
                max: 0.30,
                isPercent: true
            ),
            StockParamConfig(
                key: "tax_rate",
                label: "Effective Tax Rate",
                description: "Estimated sustainable tax rate.",
                unit: .percent,
                step: 0.005,
                min: 0.0,
                max: 0.50,
                isPercent: true
            ),
            StockParamConfig(
                key: "net_cash",
                label: "Net Cash / (Debt)",
                description: "Total Cash minus Total Debt.",
                unit: .currency,
                step: 1_000_000.0,
                min: -1_000_000_000_000.0,
                max: 1_000_000_000_000.0,
                isPercent: false
            )
        ]
    ]

    private static func clip(_ val: Double, min: Double, max: Double) -> Double {
        Swift.max(min, Swift.min(max, val))
    }

    static func calculateCustomModelValue(
        modelKey: String,
        model: IntrinsicValueResponse.Model?,
        customOverrides: [String: Double] = [:]
    ) -> Double? {
        guard let model = model else { return nil }

        // Extract base parameter dictionary
        var p: [String: Double] = [:]
        if let rawParams = model.parameters {
            for (k, v) in rawParams {
                if let d = v.doubleValue {
                    p[k] = d
                }
            }
        }
        for (k, v) in customOverrides {
            p[k] = v
        }

        switch modelKey {
        case "dcf":
            guard let baseFcf = p["base_fcf"], baseFcf > 0 else { return nil }
            let discountRate = clip(p["discount_rate"] ?? 0.09, min: 0.075, max: 0.30)
            let growthRate = p["growth_rate"] ?? 0.05
            let appliedGrowth = clip(growthRate, min: -0.05, max: 0.30)
            let terminalGrowth = p["terminal_growth_rate"] ?? 0.02
            let projectionYears = Int(round(p["projection_years"] ?? 10.0))
            let shares = p["shares_outstanding"]
            let cash = p["total_cash"] ?? 0.0
            let debt = p["total_debt"] ?? 0.0

            var pvFcf: Double = 0.0
            var currentFcf = baseFcf
            for y in 1...projectionYears {
                let fade = projectionYears > 1 ? Double(y - 1) / Double(projectionYears - 1) : 0.0
                let yearlyG = appliedGrowth - (appliedGrowth - terminalGrowth) * fade
                let nextFcf = currentFcf * (1.0 + yearlyG)
                pvFcf += nextFcf / pow(1.0 + discountRate, Double(y) - 0.5)
                currentFcf = nextFcf
            }

            let safeDiscount = max(discountRate, terminalGrowth + 0.01)
            let terminalValue = (currentFcf * (1.0 + terminalGrowth)) / (safeDiscount - terminalGrowth)
            let pvTerminal = terminalValue / pow(1.0 + discountRate, Double(projectionYears))

            let enterpriseValue = pvFcf + pvTerminal
            let equityValue = enterpriseValue + cash - debt

            if let s = shares, s > 0 {
                let iv = equityValue / s
                return iv > 0 && iv.isFinite ? round(iv * 100.0) / 100.0 : nil
            }
            if let iv = model.intrinsicValue, let base = model.parameters?["base_fcf"]?.doubleValue, base > 0 {
                let scaled = iv * (baseFcf / base)
                return round(scaled * 100.0) / 100.0
            }
            return nil

        case "dcfo":
            guard let cfoPerShare = p["cfo_per_share"], cfoPerShare > 0 else { return nil }
            let discountRate = clip(p["discount_rate"] ?? 0.09, min: 0.06, max: 0.25)
            let growthRate = p["growth_rate"] ?? 0.05
            let appliedGrowth = clip(growthRate, min: -0.05, max: 0.20)
            let terminalGrowth = p["terminal_growth_rate"] ?? 0.02
            let projectionYears = Int(round(p["projection_years"] ?? 10.0))
            let netCashPerShare = p["net_cash_per_share"] ?? 0.0

            var pvCfos: Double = 0.0
            var currentCfo = cfoPerShare
            for y in 1...projectionYears {
                let fade = projectionYears > 1 ? Double(y - 1) / Double(projectionYears - 1) : 0.0
                let yearlyG = appliedGrowth - (appliedGrowth - terminalGrowth) * fade
                let nextCfo = currentCfo * (1.0 + yearlyG)
                pvCfos += nextCfo / pow(1.0 + discountRate, Double(y) - 0.5)
                currentCfo = nextCfo
            }

            let safeDiscount = max(discountRate, terminalGrowth + 0.01)
            let terminalCfo = (currentCfo * (1.0 + terminalGrowth)) / (safeDiscount - terminalGrowth)
            let pvTerminal = terminalCfo / pow(1.0 + discountRate, Double(projectionYears))

            let iv = pvCfos + pvTerminal + netCashPerShare
            return iv > 0 && iv.isFinite ? round(iv * 100.0) / 100.0 : nil

        case "dni":
            guard let baseEps = p["base_eps"], baseEps > 0 else { return nil }
            let discountRate = clip(p["discount_rate"] ?? 0.09, min: 0.06, max: 0.25)
            let growthRate = p["growth_rate"] ?? 0.05
            let appliedGrowth = clip(growthRate, min: -0.05, max: 0.20)
            let terminalGrowth = p["terminal_growth_rate"] ?? 0.02
            let projectionYears = Int(round(p["projection_years"] ?? 10.0))

            var pvEps: Double = 0.0
            var currentEps = baseEps
            for y in 1...projectionYears {
                let fade = projectionYears > 1 ? Double(y - 1) / Double(projectionYears - 1) : 0.0
                let yearlyG = appliedGrowth - (appliedGrowth - terminalGrowth) * fade
                let nextEps = currentEps * (1.0 + yearlyG)
                pvEps += nextEps / pow(1.0 + discountRate, Double(y) - 0.5)
                currentEps = nextEps
            }

            let safeDiscount = max(discountRate, terminalGrowth + 0.01)
            let terminalVal = (currentEps * (1.0 + terminalGrowth)) / (safeDiscount - terminalGrowth)
            let pvTerminal = terminalVal / pow(1.0 + discountRate, Double(projectionYears))

            let iv = pvEps + pvTerminal
            return iv > 0 && iv.isFinite ? round(iv * 100.0) / 100.0 : nil

        case "ddm":
            guard let baseDividend = p["base_dividend"], baseDividend > 0 else { return nil }
            let discountRate = clip(p["discount_rate"] ?? 0.09, min: 0.06, max: 0.25)
            let growthRate = p["growth_rate"] ?? 0.035
            let appliedGrowth = clip(growthRate, min: -0.05, max: 0.15)
            let terminalGrowth = p["terminal_growth_rate"] ?? 0.02
            let projectionYears = Int(round(p["projection_years"] ?? 10.0))

            var pvDivs: Double = 0.0
            var currentDiv = baseDividend
            for y in 1...projectionYears {
                let fade = projectionYears > 1 ? Double(y - 1) / Double(projectionYears - 1) : 0.0
                let yearlyG = appliedGrowth - (appliedGrowth - terminalGrowth) * fade
                let nextDiv = currentDiv * (1.0 + yearlyG)
                pvDivs += nextDiv / pow(1.0 + discountRate, Double(y) - 0.5)
                currentDiv = nextDiv
            }

            let safeDiscount = max(discountRate, terminalGrowth + 0.01)
            let terminalPrice = (currentDiv * (1.0 + terminalGrowth)) / (safeDiscount - terminalGrowth)
            let pvTerminal = terminalPrice / pow(1.0 + discountRate, Double(projectionYears))

            let iv = pvDivs + pvTerminal
            return iv > 0 && iv.isFinite ? round(iv * 100.0) / 100.0 : nil

        case "mean_pe":
            guard let eps = p["eps"], eps > 0 else { return nil }
            let appliedPe = clip(p["applied_pe"] ?? 17.5, min: 5.0, max: 45.0)
            let iv = eps * appliedPe
            return iv > 0 && iv.isFinite ? round(iv * 100.0) / 100.0 : nil

        case "peg":
            guard let eps = p["eps"], eps > 0 else { return nil }
            let growthRatePct = p["growth_rate_pct"] ?? p["growth_rate"] ?? 10.0
            let divYieldPct = p["dividend_yield_pct"] ?? 0.0
            let appliedGrowth = clip(growthRatePct, min: 3.0, max: 35.0)
            let targetPeg = clip(p["target_peg"] ?? 1.0, min: 0.5, max: 2.0)
            let fairPe = clip(targetPeg * (appliedGrowth + divYieldPct), min: 5.0, max: 35.0)
            let iv = eps * fairPe
            return iv > 0 && iv.isFinite ? round(iv * 100.0) / 100.0 : nil

        case "mean_pb":
            guard let bvps = p["book_value_per_share"], bvps > 0 else { return nil }
            let appliedPb = clip(p["applied_pb"] ?? 1.5, min: 0.4, max: 8.0)
            let iv = bvps * appliedPb
            return iv > 0 && iv.isFinite ? round(iv * 100.0) / 100.0 : nil

        case "mean_ps":
            guard let sps = p["sales_per_share"], sps > 0 else { return nil }
            let appliedPs = clip(p["applied_ps"] ?? 2.5, min: 0.4, max: 15.0)
            let iv = sps * appliedPs
            return iv > 0 && iv.isFinite ? round(iv * 100.0) / 100.0 : nil

        case "psg":
            guard let sps = p["sales_per_share"], sps > 0 else { return nil }
            let revGrowthPct = p["revenue_growth_pct"] ?? p["applied_growth_pct"] ?? 15.0
            let appliedGrowth = clip(revGrowthPct, min: 5.0, max: 45.0)
            let grossMarginPct = p["gross_margin_pct"] ?? 50.0
            let gm = clip(grossMarginPct / 100.0, min: 0.20, max: 0.95)
            let targetPsg = clip(p["target_psg"] ?? 1.0, min: 0.5, max: 2.0)
            let fairPs = clip((appliedGrowth / 10.0) * gm * targetPsg * 1.5, min: 0.5, max: 12.0)
            let iv = sps * fairPs
            return iv > 0 && iv.isFinite ? round(iv * 100.0) / 100.0 : nil

        case "graham":
            guard let eps = p["eps"], eps > 0 else { return nil }
            let growthRatePct = p["growth_rate_pct"] ?? p["applied_growth_pct"] ?? 10.0
            let appliedGrowth = clip(growthRatePct, min: -5.0, max: 15.0)
            let bondYield = max(p["bond_yield_proxy"] ?? 4.5, 3.5)
            let iv = (eps * (8.5 + 2.0 * appliedGrowth) * 4.4) / bondYield
            return iv > 0 && iv.isFinite ? round(iv * 100.0) / 100.0 : nil

        case "lynch":
            guard let eps = p["eps"], eps > 0 else { return nil }
            let growthRatePct = p["growth_rate_pct"] ?? 10.0
            let divYieldPct = p["dividend_yield_pct"] ?? 0.0
            let fairMultiplier = clip(growthRatePct + divYieldPct, min: 5.0, max: 25.0)
            let iv = eps * fairMultiplier
            return iv > 0 && iv.isFinite ? round(iv * 100.0) / 100.0 : nil

        case "epv":
            guard let normEbit = p["normalized_ebit"], normEbit > 0 else { return nil }
            let discountRate = clip(p["discount_rate"] ?? 0.09, min: 0.06, max: 0.25)
            let taxRate = clip(p["tax_rate"] ?? 0.21, min: 0.0, max: 0.50)
            let netCash = p["net_cash"] ?? 0.0
            let shares = p["shares"]

            let nopat = normEbit * (1.0 - taxRate)
            let enterpriseValue = nopat / discountRate
            let equityValue = enterpriseValue + netCash

            if let s = shares, s > 0 {
                let iv = equityValue / s
                return iv > 0 && iv.isFinite ? round(iv * 100.0) / 100.0 : nil
            }
            if let iv = model.intrinsicValue, let baseEbit = model.parameters?["normalized_ebit"]?.doubleValue, baseEbit > 0 {
                let scaled = iv * (normEbit / baseEbit)
                return round(scaled * 100.0) / 100.0
            }
            return nil

        default:
            return model.intrinsicValue
        }
    }

    static func calculateBlendedScore(
        intrinsicValue: IntrinsicValueResponse?,
        customOverrides: [String: [String: Double]],
        sector: String?
    ) -> BlendedValuationResult {
        guard let iv = intrinsicValue, let models = iv.models else {
            return BlendedValuationResult(customAverage: nil, customMarginOfSafety: nil, customModelValues: [:], hasAnyCustom: false)
        }

        var customValues: [String: Double] = [:]
        var hasAnyCustom = false

        let modelMap: [String: IntrinsicValueResponse.Model?] = [
            "dcf": models.dcf,
            "dcfo": models.dcfo,
            "dni": models.dni,
            "mean_pe": models.meanPe,
            "peg": models.peg,
            "mean_pb": models.meanPb,
            "mean_ps": models.meanPs,
            "psg": models.psg,
            "graham": models.graham,
            "ddm": models.ddm,
            "epv": models.epv,
            "lynch": models.lynch,
        ]

        for (k, m) in modelMap {
            guard let model = m else { continue }
            let overrides = customOverrides[k] ?? [:]
            let defaultParams = model.parameters ?? [:]
            let isActuallyModified = overrides.contains { (paramKey, val) in
                if let def = defaultParams[paramKey]?.doubleValue {
                    return abs(val - def) > 1e-5
                }
                return true
            }

            if isActuallyModified {
                hasAnyCustom = true
                if let customVal = calculateCustomModelValue(modelKey: k, model: model, customOverrides: overrides) {
                    customValues[k] = customVal
                }
            } else if let origVal = model.intrinsicValue {
                customValues[k] = origVal
            }
        }

        if !hasAnyCustom {
            return BlendedValuationResult(
                customAverage: iv.averageIntrinsicValue,
                customMarginOfSafety: iv.marginOfSafetyPct,
                customModelValues: customValues,
                hasAnyCustom: false
            )
        }

        // The backend's own weights: they encode which models may describe this
        // business *and* how tightly each pinned its answer (reliability from
        // its Monte Carlo band), neither of which can be recomputed here.
        // Editing a parameter changes what a model says, never whether it votes.
        // The priors below are only a fallback for a cached response sent
        // before the API carried weights; `sector` serves the same purpose.
        let weights: [String: Double]
        if let shipped = iv.modelWeights, !shipped.isEmpty {
            weights = shipped
        } else {
            let haystack = (iv.blendProfile ?? sector ?? "").lowercased()
            if haystack.contains("reit") {
                weights = ["dcfo": 0.50, "ddm": 0.50]
            } else if ["financial", "bank", "insurance"].contains(where: { haystack.contains($0) }) {
                weights = ["dni": 0.85, "ddm": 0.15]
            } else {
                weights = ["dcf": 0.55, "graham": 0.30, "ddm": 0.15]
            }
        }

        var totalWeight: Double = 0.0
        var weightedSum: Double = 0.0

        for (k, w) in weights where w > 0 {
            if let val = customValues[k], val > 0, val.isFinite {
                weightedSum += val * w
                totalWeight += w
            }
        }

        var customAvg: Double? = nil
        if totalWeight > 0 {
            var blended = weightedSum / totalWeight
            // The same credible band the backend clamps to, so a slider cannot
            // produce a headline the API would have refused to publish.
            if let spot = iv.currentPrice, spot > 0 {
                blended = min(max(blended, 0.1 * spot), 5.0 * spot)
            }
            customAvg = round(blended * 100.0) / 100.0
        } else if let first = customValues.values.first {
            customAvg = first
        }

        var customMos: Double? = nil
        if let avg = customAvg, let spot = iv.currentPrice, spot > 0 {
            customMos = round(((avg - spot) / spot) * 1000.0) / 10.0
        }

        return BlendedValuationResult(
            customAverage: customAvg,
            customMarginOfSafety: customMos,
            customModelValues: customValues,
            hasAnyCustom: true
        )
    }
}
