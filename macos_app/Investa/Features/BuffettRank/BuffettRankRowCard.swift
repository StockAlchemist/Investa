import SwiftUI

/// Formatting and colour shared by every figure in the ranking.
///
/// The five pillars, the quality and value halves and the composite are all
/// winsorised percentiles on one 0–100 scale, so a single ramp reads them all:
/// a 70 means the same thing in the Growth column as it does in the composite.
enum BuffettScore {
    static func text(_ value: Double?) -> String {
        guard let value else { return "—" }
        return String(format: "%.0f", value)
    }

    static func tint(_ value: Double?) -> Color {
        guard let value else { return .secondary }
        if value >= 70 { return .up }
        if value >= 50 { return .brandCyan }
        if value >= 30 { return .brandAmber }
        return .down
    }

    /// The meter fill, clamped — a percentile cannot leave the track.
    static func fraction(_ value: Double?) -> Double {
        min(max((value ?? 0) / 100, 0), 1)
    }
}

/// One ranked company.
///
/// The card is read in two passes and is built that way: an identity line that
/// says which company and how it scored, then the evidence for that score — the
/// five quality percentiles as meters, and the two yields the value half is
/// actually made of. The meters are the point: fifty rows of bare two-digit
/// numbers have no shape, and the shape is what makes a long list scannable.
struct BuffettRankRowCard: View {
    let row: BuffettRankRow
    let onOpen: () -> Void

    var body: some View {
        Button(action: onOpen) {
            Group {
                if isPhoneLayout {
                    // No rule and tighter spacing: on a 400pt screen the divider
                    // and the air around it cost a fifth of a row for a line
                    // that separates two things already separated by their type.
                    VStack(alignment: .leading, spacing: 7) {
                        phoneIdentity
                        phoneSummary
                        pillarStrip
                    }
                } else {
                    VStack(alignment: .leading, spacing: 10) {
                        identity
                        Divider().opacity(0.4)
                        metrics
                    }
                }
            }
            .padding(isPhoneLayout ? 10 : 12)
            .frame(maxWidth: .infinity, alignment: .leading)
            // The composite's colour, carried down the card's edge. It is what
            // lets the eye find the strong companies while scrolling past the
            // ones it isn't reading.
            .overlay(alignment: .leading) {
                Rectangle()
                    .fill(BuffettScore.tint(row.compositeScore))
                    .frame(width: 3)
                    .allowsHitTesting(false)
            }
            .card()
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    // MARK: - Identity

    private var identity: some View {
        HStack(spacing: 10) {
            Text(row.rank.map(String.init) ?? "—")
                .appFont(.callout.monospacedDigit().weight(.semibold))
                .foregroundStyle((row.rank ?? .max) <= 3 ? Color.brand : Color.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.6)
                .frame(width: 30, alignment: .trailing)

            // The logo is what makes a hundred-row list scannable: a company is
            // recognised by its mark long before its ticker is read.
            StockIcon(symbol: row.symbol, size: isPhoneLayout ? 30 : 34, scalesWithText: true)

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 5) {
                    Text(row.symbol)
                        .appFont(.headline)
                        .lineLimit(1)
                        .minimumScaleFactor(0.7)
                    if row.model != .generic {
                        BuffettTagBadge(text: row.model.shortLabel)
                    }
                    // Flagged beside the identity rather than the score: a
                    // company demoted for thin filings should not look like one
                    // that is honestly mediocre.
                    if row.isConfidenceReduced {
                        BuffettTagBadge(text: "THIN DATA", tint: .brandAmber)
                    }
                    // Same warning the stock page shows. A ranked row is
                    // something a reader may act on, and the value half of the
                    // score is computed from exactly this price series.
                    if row.dataQuality != nil {
                        BuffettTagBadge(text: "PRICE DATA", tint: .brandAmber)
                    }
                }
                if let name = row.name, !name.isEmpty {
                    // Two lines rather than one: at phone width a single line
                    // ends in an ellipsis for most of the market.
                    Text(name)
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            Spacer(minLength: 8)

            // Scale, which the ranking itself is blind to: a 90 from a $200M
            // bank and a 90 from a mega-cap are the same score about very
            // different propositions. Dropped on phone, where the row has no
            // room for a third column.
            if !isPhoneLayout, let cap = row.marketCap, cap > 0 {
                VStack(alignment: .trailing, spacing: 1) {
                    Text(Fmt.compact(cap, code: "USD"))
                        .appFont(.callout.monospacedDigit())
                        .foregroundStyle(.secondary)
                    Text("Mkt cap")
                        .appFont(.caption2)
                        .foregroundStyle(.tertiary)
                }
                .lineLimit(1)
                .minimumScaleFactor(0.7)
            }

            BuffettScoreBadge(
                composite: row.compositeScore,
                quality: row.qualityScore,
                value: row.valueScore
            )
            .layoutPriority(1)
        }
    }

    // MARK: - Phone

    /// The same row, rebuilt for a screen a quarter the width.
    ///
    /// The two halves move off the identity line and into the summary beneath
    /// it. That is worth ~35pt of width, which is the difference between a
    /// company name fitting on its line and wrapping onto a second — and a
    /// wrapped name is the single biggest thing standing between the reader and
    /// the next row.
    private var phoneIdentity: some View {
        HStack(spacing: 8) {
            Text(row.rank.map(String.init) ?? "—")
                .appFont(.callout.monospacedDigit().weight(.semibold))
                .foregroundStyle((row.rank ?? .max) <= 3 ? Color.brand : Color.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.6)
                .frame(width: 24, alignment: .trailing)

            StockIcon(symbol: row.symbol, size: 28, scalesWithText: true)

            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 5) {
                    Text(row.symbol)
                        .appFont(.headline)
                        .lineLimit(1)
                        .minimumScaleFactor(0.7)
                    if row.model != .generic {
                        BuffettTagBadge(text: row.model.shortLabel)
                    }
                    if row.isConfidenceReduced {
                        BuffettTagBadge(text: "THIN", tint: .brandAmber)
                    }
                    // The compact layout keeps the label short for the same
                    // reason its neighbour is "THIN" and not "THIN DATA":
                    // at phone width a longer badge pushes the row.
                    if row.dataQuality != nil {
                        BuffettTagBadge(text: "PRICE", tint: .brandAmber)
                    }
                }
                if let name = row.name, !name.isEmpty {
                    Text(name)
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            Spacer(minLength: 6)

            BuffettCompositeTile(score: row.compositeScore)
        }
    }

    /// Everything that was a chip on Mac, on one line: the two halves of the
    /// composite on the left, the two yields on the right. Four short tokens
    /// cost 14pt here where two rows of chips cost 55.
    private var phoneSummary: some View {
        HStack(spacing: 5) {
            BuffettHalfPill(prefix: "Q", score: row.qualityScore)
            BuffettHalfPill(prefix: "V", score: row.valueScore)
            Spacer(minLength: 6)
            yieldText("E/P", row.earningsYield, isScored: true)
            yieldText("FCF/P", row.fcfYield, isScored: row.scoresFcfYield)
        }
        .lineLimit(1)
        .minimumScaleFactor(0.6)
    }

    private func yieldText(_ label: String, _ value: Double?, isScored: Bool) -> some View {
        HStack(spacing: 3) {
            Text(label)
                .appFont(.caption2)
                .foregroundStyle(.secondary)
            Text(isScored ? Fmt.percent(value) : "n/a")
                .appFont(.caption.monospacedDigit().weight(.semibold))
                .foregroundStyle(yieldTint(value, isScored: isScored))
        }
    }

    private func yieldTint(_ value: Double?, isScored: Bool) -> Color {
        guard isScored, let value else { return Color.secondary.opacity(0.7) }
        return value > 0 ? .up : .down
    }

    // MARK: - Evidence

    /// Quality and value, kept visually apart because they are measured
    /// differently: the pillars are percentiles against the company's own model
    /// and carry meters, the yields are the actual figures and do not.
    private var metrics: some View {
        HStack(spacing: 14) {
            pillarStrip
            Rectangle()
                .fill(Color.secondary.opacity(0.18))
                .frame(width: 1, height: 34)
            yieldChips
        }
    }

    private var pillarStrip: some View {
        HStack(spacing: 8) {
            ForEach(row.pillars, id: \.label) { pillar in
                BuffettPillarCell(label: pillarLabel(pillar.label), value: pillar.value)
            }
        }
        .frame(maxWidth: .infinity)
    }

    /// The two inputs the value half is built from — 60% earnings yield, 40%
    /// free-cash-flow yield. Both keep their place on every card rather than
    /// dropping out, so a column can be scanned down the list without being
    /// re-found on each row; a bank, whose model has no free-cash-flow yield to
    /// give, reads "n/a" rather than a dash.
    @ViewBuilder
    private var yieldChips: some View {
        BuffettYieldChip(label: "E/P", value: row.earningsYield)
        BuffettYieldChip(label: "FCF/P", value: row.fcfYield, isScored: row.scoresFcfYield)
    }

    /// Full labels on Mac, abbreviated where a phone column cannot hold one.
    private func pillarLabel(_ label: String) -> String {
        guard isPhoneLayout else { return label }
        return label == "Predictable" ? "Predict." : label
    }
}

// MARK: - Score badge

/// The composite, with the two halves it was blended from underneath.
struct BuffettScoreBadge: View {
    let composite: Double?
    let quality: Double?
    let value: Double?

    private var tone: Color { BuffettScore.tint(composite) }

    var body: some View {
        VStack(alignment: .trailing, spacing: 4) {
            BuffettCompositeTile(score: composite)
            HStack(spacing: 4) {
                BuffettHalfPill(prefix: "Q", score: quality)
                BuffettHalfPill(prefix: "V", score: value)
            }
        }
        .lineLimit(1)
        .minimumScaleFactor(0.6)
    }
}

/// The composite, boxed in its own colour.
struct BuffettCompositeTile: View {
    let score: Double?

    private var tone: Color { BuffettScore.tint(score) }

    var body: some View {
        Text(BuffettScore.text(score))
            .appFont(.title3.monospacedDigit().weight(.bold))
            .foregroundStyle(tone)
            .lineLimit(1)
            .minimumScaleFactor(0.6)
            .frame(minWidth: 34)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(tone.opacity(0.12), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(tone.opacity(0.28), lineWidth: 1)
            )
    }
}

/// One half of the composite — quality or value — at its own tint.
struct BuffettHalfPill: View {
    let prefix: String
    let score: Double?

    var body: some View {
        let tint = BuffettScore.tint(score)
        return Text("\(prefix) \(BuffettScore.text(score))")
            .appFont(.caption.monospacedDigit().weight(.semibold))
            .foregroundStyle(tint)
            .lineLimit(1)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(tint.opacity(0.10), in: Capsule())
    }
}

// MARK: - Pillar cell

/// One quality percentile: the figure, a meter of it, and what it measures.
struct BuffettPillarCell: View {
    let label: String
    let value: Double?

    private var tone: Color { BuffettScore.tint(value) }
    /// A ceiling, not a width — see `meter`. Generous on Mac, where a cell in a
    /// wide window is ~200pt and a 52pt stub under the figure read as an
    /// underline rather than as a measurement.
    private var track: CGFloat { isPhoneLayout ? 40 : 120 }

    var body: some View {
        VStack(spacing: 3) {
            Text(BuffettScore.text(value))
                .appFont(.callout.monospacedDigit().weight(.semibold))
                .foregroundStyle(tone)
            meter
            Text(label)
                .appFont(.caption2)
                .foregroundStyle(.secondary)
        }
        .lineLimit(1)
        .minimumScaleFactor(0.6)
        .frame(maxWidth: .infinity)
    }

    /// `maxWidth` rather than `width`, and a scale rather than a measured fill.
    ///
    /// A pinned width would turn "prefers 52pt" into "demands 52pt" — five of
    /// them per row is then a card that asks for more width than a narrow split
    /// view can offer, which is how a page ends up scrolled off its own right
    /// edge. `scaleEffect` is a render-time transform, so the fill needs no
    /// `GeometryReader` and costs the layout nothing.
    private var meter: some View {
        Capsule()
            .fill(Color.secondary.opacity(0.18))
            .frame(maxWidth: track)
            .frame(height: 3.5)
            .overlay(alignment: .leading) {
                Capsule()
                    .fill(tone)
                    .scaleEffect(x: BuffettScore.fraction(value), anchor: .leading)
            }
            .opacity(value == nil ? 0.3 : 1)
    }
}

// MARK: - Yield chip

/// An actual yield, not a percentile — so it carries no meter and no ramp,
/// only the sign that says whether the company earned anything at all.
struct BuffettYieldChip: View {
    let label: String
    /// Already a percentage from the backend (10.4 → "10.40%").
    let value: Double?
    /// False where this company's model never computes the metric at all. The
    /// difference matters: "—" says the figure went missing, "n/a" says the
    /// value score was never built from it.
    var isScored: Bool = true

    private var tone: Color {
        guard isScored else { return .secondary }
        guard let value else { return .secondary }
        return value > 0 ? .up : .down
    }

    var body: some View {
        VStack(spacing: 2) {
            Text(isScored ? Fmt.percent(value) : "n/a")
                .appFont(.callout.monospacedDigit().weight(.semibold))
                .foregroundStyle(isScored ? tone : Color.secondary.opacity(0.7))
            Text(label)
                .appFont(.caption2)
                .foregroundStyle(.secondary)
        }
        .lineLimit(1)
        .minimumScaleFactor(0.6)
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 9, style: .continuous))
    }
}

// MARK: - Badge

/// Small uppercase tag — the valuation model a company was scored under, or a
/// warning about the data behind its score.
struct BuffettTagBadge: View {
    let text: String
    var tint: Color = .secondary

    var body: some View {
        Text(text)
            .appFont(.system(size: 9, weight: .heavy))
            .tracking(0.6)
            .foregroundStyle(tint == .secondary ? Color.secondary : tint)
            .lineLimit(1)
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .background(tint.opacity(tint == .secondary ? 0.14 : 0.15), in: Capsule())
    }
}
