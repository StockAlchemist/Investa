import SwiftUI

/// How many x labels a continuous chart can carry, and which ones.
///
/// `AxisMarks(values: .automatic(desiredCount: n))` is a *hint*, and on a date
/// or numeric scale Swift Charts routinely ignores it: it rounds to a "nice"
/// interval — a year, two years, an hour, 10⁵ seconds — and then emits every
/// tick the domain contains. A fourteen-year position history answers a request
/// for three labels with eight. Axis labels here carry `.fixedSize()` (they
/// must — an axis label is offered only its own tick's slot, so without it a
/// date truncates to `Jun 2…`), so the surplus can't shrink out of the way: it
/// overprints, and `Dec 2010Dec 2012Dec 2014` is not a date.
///
/// So the ticks are picked here instead, from the data, against the width the
/// chart was actually offered. Same bargain `PeriodChartMetrics` makes for the
/// category charts: approximate on purpose, because these pick a label count
/// and never a frame, so a measurement that lags a layout pass can only
/// mis-round.
enum ChartAxis {
    /// What is spent before the first x label is drawn: the card's own padding
    /// on both sides, the y labels, and the plot's trailing inset. `width` is
    /// the width the *card* was offered, so all of it comes off — 60 counted
    /// the y axis and forgot the padding, which is a third of the shortfall
    /// that put four touching labels on a 402pt phone.
    static let axisOverhead: CGFloat = 80

    /// Mean advance of a `.caption2` glyph, as a fraction of the point size.
    /// Measured off a rendered axis: "Dec 2025" is ~58pt at 11pt, and the
    /// digits a date label is mostly made of are wider than the 0.62 this
    /// started at.
    private static let glyphWidth: CGFloat = 0.66

    /// The clear space between two labels, below which they read as one word.
    private static let labelGap: CGFloat = 14

    /// The room one axis label needs: the string at `.caption2`, plus the gap
    /// that keeps two of them from reading as one word.
    ///
    /// `sample` is the *widest* label the axis will draw — `"Dec 2010"`, not
    /// `"1 Jan"` — because capacity has to be budgeted for the longest one.
    /// Budget for the reader's type size too, not the author's: at an
    /// accessibility size every label is ~1.4× wider, which is exactly the
    /// frame where an axis that "fits" starts overprinting.
    static func labelWidth(
        _ sample: String,
        scale: CGFloat = 1,
        typeSize: DynamicTypeSize = .large
    ) -> CGFloat {
        textWidth(sample, scale: scale, typeSize: typeSize) + labelGap
    }

    /// The label's ink alone, without the gap beside it.
    private static func textWidth(
        _ sample: String,
        scale: CGFloat,
        typeSize: DynamicTypeSize
    ) -> CGFloat {
        CGFloat(sample.count) * captionPointSize * textScale(scale: scale, typeSize: typeSize) * glyphWidth
    }

    /// How much wider text is here than at the author's own settings — the
    /// window's `appFontScale` and the reader's Dynamic Type size together.
    /// For the width budgets that predate this file (`PeriodChartMetrics`),
    /// which measure in points rather than characters.
    static func textScale(scale: CGFloat = 1, typeSize: DynamicTypeSize = .large) -> CGFloat {
        scale * typeScale(typeSize)
    }

    /// How many labels of `labelWidth` fit across a chart offered `width`, on a
    /// **band** axis — one label centred under each category, so they share the
    /// plot evenly and the count is simply how many go into it.
    ///
    /// `unmeasured` is what the first frame gets, before any measurement lands:
    /// the count a phone can carry, so the opening frame is never the
    /// overprinted one.
    static func labelCapacity(
        width: CGFloat,
        labelWidth: CGFloat,
        unmeasured: Int = 3,
        cap: Int = 6
    ) -> Int {
        guard width > 0, labelWidth > 0 else { return unmeasured }
        return max(2, min(cap, Int((width - axisOverhead) / labelWidth)))
    }

    /// How many ticks fit across a **continuous** axis — a date or numeric
    /// scale whose first and last labels hang inward from the plot's edges.
    ///
    /// Not the same sum as `labelCapacity`, and the difference is what left
    /// four touching labels on a 1Y chart. `anchor(_:in:)` pins the outermost
    /// labels *inside* the plot rather than centring them on their ticks — the
    /// only way to keep them from being clipped — so each end label sits half
    /// its own width closer to its neighbour than the tick spacing suggests.
    /// The spacing that has to clear a label is therefore one and a half of
    /// them, not one; budget for one and the ends collide while the middle of
    /// the axis stands empty, which is exactly how the bug looked.
    static func tickCapacity(
        _ sample: String,
        width: CGFloat,
        scale: CGFloat = 1,
        typeSize: DynamicTypeSize = .large,
        unmeasured: Int = 3,
        cap: Int = 6
    ) -> Int {
        guard width > 0 else { return unmeasured }
        let plot = width - axisOverhead
        let spacing = textWidth(sample, scale: scale, typeSize: typeSize) * 1.5 + labelGap
        guard plot > 0, spacing > 0 else { return 2 }
        return max(2, min(cap, 1 + Int(plot / spacing)))
    }

    /// `count` values spread evenly across `values`, keeping both ends.
    ///
    /// Both ends, because the first and last are the two labels a reader looks
    /// for — "since when" and "as of when" — and an interval-based tick strategy
    /// drops precisely those.
    static func ticks<T: Equatable>(_ values: [T], count: Int) -> [T] {
        guard count > 0, !values.isEmpty else { return [] }
        guard values.count > count else { return values }
        guard count > 1, let last = values.last else { return [values[values.count / 2]] }

        let maxIndex = Double(values.count - 1)
        var out: [T] = []
        for i in 0..<count {
            let idx = Int((Double(i) / Double(count - 1) * maxIndex).rounded())
            let value = values[idx]
            // A series may repeat an x value; two marks on one tick draw twice.
            if out.last != value { out.append(value) }
        }
        if out.last != last { out.append(last) }
        return out
    }

    /// Whether a run of months should be named by their initials — `J F M A M
    /// J J A S O N D` — instead of `Jan Feb Mar`.
    ///
    /// The choice is between two ways of losing information, and one of them is
    /// worse: thinning drops half the months, and a bar with no label under it
    /// is a bar the reader has to count back to. Initials keep a label under
    /// every bar, and a reader who can see `J F M A M` in order can name any of
    /// them — which is only true while the run is *complete*, so this asks for
    /// exactly that: the three-letter names no longer all fit, and the initials
    /// still do. Where even initials have to be thinned, `Mar` is the better
    /// label and this stays false.
    static func prefersMonthInitials(
        count: Int,
        width: CGFloat,
        scale: CGFloat = 1,
        typeSize: DynamicTypeSize = .large
    ) -> Bool {
        guard count > 1 else { return false }
        func fits(_ sample: String) -> Bool {
            labelCapacity(
                width: width,
                labelWidth: labelWidth(sample, scale: scale, typeSize: typeSize),
                unmeasured: 3,
                cap: count
            ) >= count
        }
        return !fits("Mar") && fits("M")
    }

    /// Where a tick label sits relative to its mark: inward at the two ends,
    /// centred everywhere else.
    ///
    /// A label centred on the first or last tick hangs half its width past the
    /// plot, where it is clipped — the trailing one is the newest date, the one
    /// worst affordable to lose.
    static func anchor<T: Equatable>(_ value: T, in ticks: [T]) -> UnitPoint {
        guard ticks.count > 1 else { return .top }
        if value == ticks.first { return .topLeading }
        if value == ticks.last { return .topTrailing }
        return .top
    }

    /// `.caption2` as the platform resolves it, before `appFontScale`. Not
    /// private, so a test can state which platform's frame it is pinning: the
    /// suite hosts on macOS, where this is 10pt, and the phone frames these
    /// budgets were measured on resolve 11.
    static var captionPointSize: CGFloat {
        #if os(iOS)
        return 11
        #else
        return 10
        #endif
    }

    /// Roughly how much wider a label gets at each Dynamic Type size. The small
    /// text styles scale less than `.body`, which is why the accessibility
    /// sizes here fall short of the full ~2.5× `.body` takes.
    private static func typeScale(_ size: DynamicTypeSize) -> CGFloat {
        switch size {
        case .xSmall: return 0.85
        case .small: return 0.9
        case .medium: return 0.95
        case .large: return 1.0
        case .xLarge: return 1.1
        case .xxLarge: return 1.2
        case .xxxLarge: return 1.3
        case .accessibility1: return 1.5
        case .accessibility2: return 1.7
        case .accessibility3: return 1.9
        case .accessibility4: return 2.0
        case .accessibility5: return 2.1
        @unknown default: return 1.0
        }
    }
}
