import SwiftUI

/// What the ranking is, and how much of the market survived it.
///
/// The run counts used to sit in the header as three bare numbers, which said
/// nothing about the shape of the thing: four fifths of the listed market fails
/// a gate, and that proportion *is* the story of the screen. So the counts are
/// drawn as the split they describe, with the method behind them available in
/// place rather than in a doc nobody opens.
struct BuffettRankHero: View {
    let run: BuffettRankRun?
    @State private var showingMethod = false

    private var ranked: Double { Double(run?.rankedCount ?? 0) }
    private var excluded: Double { Double(run?.excludedCount ?? 0) }
    private var scored: Double { max(ranked + excluded, 1) }

    var body: some View {
        VStack(alignment: .leading, spacing: isPhoneLayout ? 9 : 12) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Buffett & Value Ranking")
                    .appFont(.title2.weight(.bold))
                // Three lines of explanation ahead of the first company is a
                // fair trade on a Mac window and a bad one on a phone, where it
                // is a tenth of the screen every time the tab is opened. There
                // it moves inside "How it's scored", one tap away.
                if !isPhoneLayout {
                    Text(Self.intro)
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            if run != nil {
                splitBar
                legend
            }

            methodToggle
            if showingMethod { BuffettMethodNote(includesIntro: isPhoneLayout) }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(isPhoneLayout ? 12 : 16)
        .card(.hero)
    }

    static let intro = """
        Every US-listed common stock, scored 60% on business quality and 40% on value. \
        Quality gates run first — a company that fails one is excluded rather than ranked \
        low, because cheapness never rescues a broken business.
        """

    // MARK: - Split

    /// Ranked against excluded, at the proportion the run actually produced.
    private var splitBar: some View {
        GeometryReader { geo in
            HStack(spacing: 2) {
                Capsule()
                    .fill(LinearGradient(colors: [.brand, .brandCyan], startPoint: .leading, endPoint: .trailing))
                    .frame(width: max(3, (geo.size.width - 2) * ranked / scored))
                Capsule()
                    .fill(Color.secondary.opacity(0.22))
            }
        }
        .frame(height: 8)
        .accessibilityHidden(true)
    }

    /// Wrapped rather than scrolled: three counts that must each stay readable
    /// at any width, and a hidden count is a count nobody reads.
    private var legend: some View {
        WrappingRow(spacing: isPhoneLayout ? 12 : 16, lineSpacing: 6) {
            statistic("Ranked", run?.rankedCount, dot: .brand)
            statistic("Excluded", run?.excludedCount, dot: Color.secondary.opacity(0.5))
            statistic("Universe", run?.universeSize, dot: nil)
            if let finished = run?.finishedAt, !finished.isEmpty {
                statistic("Run", text: MarketTime.formatted(finished), dot: nil)
            }
        }
    }

    private func statistic(_ label: String, _ value: Int?, dot: Color?) -> some View {
        statistic(label, text: value.map { Fmt.number(Double($0), fractionDigits: 0) } ?? "—", dot: dot)
    }

    private func statistic(_ label: String, text: String, dot: Color?) -> some View {
        HStack(spacing: 6) {
            if let dot {
                Circle().fill(dot).frame(width: 7, height: 7)
            }
            VStack(alignment: .leading, spacing: 1) {
                Text(label)
                    .appFont(.caption2)
                    .foregroundStyle(.secondary)
                Text(text)
                    .appFont(isPhoneLayout ? .subheadline.monospacedDigit().weight(.semibold)
                                           : .headline.monospacedDigit())
            }
        }
        .lineLimit(1)
        .minimumScaleFactor(0.7)
    }

    // MARK: - Method

    private var methodToggle: some View {
        Button {
            withAnimation(.easeInOut(duration: 0.18)) { showingMethod.toggle() }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: showingMethod ? "chevron.down" : "chevron.right")
                    .appFont(.system(size: 9, weight: .bold))
                Text("How it's scored")
                    .appFont(.caption.weight(.semibold))
            }
            .foregroundStyle(Color.brand)
        }
        .buttonStyle(.plain)
    }
}

/// The weights, spelled out. They are the whole argument for a company's
/// position, and a column called "Capital" means nothing without them.
struct BuffettMethodNote: View {
    /// Set where the hero could not afford the paragraph on its own.
    var includesIntro = false

    private let quality = [
        ("Returns on capital", "30"), ("Financial strength", "20"), ("Predictability", "20"),
        ("Growth", "15"), ("Capital allocation", "15"),
    ]
    private let value = [("Earnings yield", "60"), ("Free-cash-flow yield", "40")]

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if includesIntro {
                Text(BuffettRankHero.intro)
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            group("Quality — 60% of the composite", quality, tint: .brand)
            group("Value — 40%", value, tint: .brandIndigo)
            Text("""
                 Every figure is a percentile against the companies scored under the same model, so \
                 a bank's leverage is judged against other banks and never against an industrial. \
                 Missing filings lower a company's confidence — and with it its score — rather than \
                 failing it outright. Banks and insurers have no owner-earnings figure to derive a \
                 free-cash-flow yield from, so their value score is the earnings yield alone and \
                 their FCF/P reads "n/a".
                 """)
                .appFont(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.secondary.opacity(0.07), in: RoundedRectangle(cornerRadius: Theme.insetRadius, style: .continuous))
    }

    private func group(_ title: String, _ items: [(String, String)], tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            SectionLabel(title: title)
            WrappingRow(spacing: 8, lineSpacing: 6) {
                ForEach(items, id: \.0) { item in
                    HStack(spacing: 5) {
                        Text(item.0)
                            .appFont(.caption)
                            .foregroundStyle(.secondary)
                        Text(item.1)
                            .appFont(.caption.monospacedDigit().weight(.bold))
                            .foregroundStyle(tint)
                    }
                    .lineLimit(1)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(tint.opacity(0.10), in: Capsule())
                }
            }
        }
    }
}

// MARK: - Controls

/// The two lists, the model filter and the search box.
struct BuffettRankControls: View {
    @ObservedObject var viewModel: BuffettRankViewModel

    private var trimmedSearch: String {
        viewModel.search.trimmingCharacters(in: .whitespaces)
    }

    /// The width the single-row form needs before it is worth taking. The chips
    /// wrap inside the row rather than forcing it wider, so this only has to
    /// cover the picker, one line of chips and the search box.
    private static let singleRowWidth: CGFloat = 760

    @State private var containerWidth: CGFloat = 0

    var body: some View {
        Group {
            if isPhoneLayout || prefersStackedLayout(measuredWidth: containerWidth, needs: Self.singleRowWidth) {
                stackedControls
            } else {
                // A Mac window has the width for a toolbar and no use for three
                // left-aligned rows with a page of empty space beside them.
                // Same idiom as `GlobalControlBar`: one line, trailing controls
                // pushed out by a spacer, no card of its own.
                wideControls
            }
        }
        .readingContainerWidth { containerWidth = $0 }
    }

    private var wideControls: some View {
        HStack(spacing: 12) {
            listPicker
                .frame(maxWidth: 220)

            if !viewModel.showingExclusions {
                Rectangle()
                    .fill(Color.secondary.opacity(0.18))
                    .frame(width: 1, height: 20)
                modelChips
            }

            Spacer(minLength: 12)

            if !trimmedSearch.isEmpty { matchCount }
            searchBox.frame(maxWidth: 300)
        }
    }

    private var stackedControls: some View {
        VStack(alignment: .leading, spacing: 10) {
            listPicker
            if isPhoneLayout {
                // Five chips wrap onto two rows at phone width and cost ~90pt
                // before a single company is on screen, so the filter becomes a
                // menu and shares its line with the search box. A menu also
                // states the current filter in words, which a row of chips only
                // does by fill colour.
                HStack(spacing: 8) {
                    searchField
                    if !viewModel.showingExclusions { modelMenu }
                }
            } else {
                if !viewModel.showingExclusions { modelChips }
                searchField
            }
        }
    }

    private var modelMenu: some View {
        Menu {
            Picker("Model", selection: Binding(
                get: { viewModel.model },
                set: { newModel in Task { await viewModel.select(model: newModel) } }
            )) {
                Text("All models").tag(BuffettModel?.none)
                ForEach(BuffettModel.allCases) { model in
                    Text(model.label).tag(BuffettModel?.some(model))
                }
            }
        } label: {
            HStack(spacing: 4) {
                Text(viewModel.model?.label ?? "All models")
                    .appFont(.caption.weight(.semibold))
                Image(systemName: "chevron.up.chevron.down")
                    .appFont(.system(size: 8, weight: .bold))
            }
            .lineLimit(1)
            .minimumScaleFactor(0.7)
            .foregroundStyle(viewModel.model == nil ? Color.primary : Color.white)
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(viewModel.model == nil ? Color.secondary.opacity(0.12) : Color.brand, in: Capsule())
        }
    }

    private var listPicker: some View {
        Picker("List", selection: Binding(
            get: { viewModel.showingExclusions },
            set: { showing in Task { await viewModel.setShowingExclusions(showing) } }
        )) {
            Text("Ranked").tag(false)
            Text("Excluded").tag(true)
        }
        .pickerStyle(.segmented)
        .labelsHidden()
        .frame(maxWidth: 280)
    }

    private var modelChips: some View {
        WrappingRow(spacing: 8, lineSpacing: 8) {
            BuffettFilterChip(label: "All models", isSelected: viewModel.model == nil) {
                Task { await viewModel.select(model: nil) }
            }
            ForEach(BuffettModel.allCases) { model in
                BuffettFilterChip(label: model.label, isSelected: viewModel.model == model) {
                    Task { await viewModel.select(model: model) }
                }
            }
        }
    }

    /// Serves both lists: when a company is missing from the ranking, looking it
    /// up in the excluded list is the very next thing anyone does. The search
    /// runs server-side over the whole run — filtering the loaded rows would
    /// only ever search what is already on screen.
    private var searchField: some View {
        HStack(spacing: 8) {
            searchBox.frame(maxWidth: 340)
            if !trimmedSearch.isEmpty { matchCount }
        }
    }

    private var matchCount: some View {
        Text("\(viewModel.totalMatches) match\(viewModel.totalMatches == 1 ? "" : "es")")
            .appFont(.caption)
            .foregroundStyle(.secondary)
            .lineLimit(1)
            .minimumScaleFactor(0.7)
    }

    private var searchBox: some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .appFont(.caption)
                .foregroundStyle(.secondary)
            TextField(viewModel.showingExclusions ? "Search excluded stocks…" : "Search all ranked stocks…",
                      text: $viewModel.search)
                .textFieldStyle(.plain)
                .appFont(.callout)
                .autocorrectionDisabled()
                .onChange(of: viewModel.search) { _, _ in viewModel.searchChanged() }
            if !trimmedSearch.isEmpty {
                Button {
                    viewModel.search = ""
                    viewModel.searchChanged()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(Color.secondary.opacity(0.10), in: Capsule())
    }
}

/// A filter pill. Selected reads as filled rather than merely tinted, because a
/// row of chips where only the opacity differs does not say which one is on.
struct BuffettFilterChip: View {
    let label: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(label)
                .appFont(.caption.weight(.semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.8)
                .foregroundStyle(isSelected ? Color.white : Color.primary)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(isSelected ? Color.brand : Color.secondary.opacity(0.12), in: Capsule())
                .overlay(
                    Capsule().strokeBorder(isSelected ? Color.clear : Color.secondary.opacity(0.22), lineWidth: 1)
                )
        }
        .buttonStyle(.plain)
    }
}
