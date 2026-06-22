import SwiftUI

@MainActor
final class AIViewModel: ObservableObject {
    @Published var review: PortfolioAIReview?
    @Published var isLoading = false
    @Published var isRefreshing = false
    @Published var errorMessage: String?

    private let api: APIClient
    init(api: APIClient = .shared) { self.api = api }

    func load(currency: String, accounts: [String]?, refresh: Bool) async {
        if refresh { isRefreshing = true } else { isLoading = true }
        errorMessage = nil
        defer { isLoading = false; isRefreshing = false }
        do {
            review = try await api.send(
                method: "POST", path: "/portfolio/ai_review",
                query: [URLQueryItem(name: "currency", value: currency),
                        URLQueryItem(name: "refresh", value: refresh ? "true" : "false")]
                    + APIClient.arrayQuery("accounts", accounts))
        } catch let error as APIError {
            errorMessage = error.errorDescription
        } catch { errorMessage = error.localizedDescription }
    }
}

private struct MetricDetail: Identifiable { let id = UUID(); let title: String; let score: Double; let content: String }

struct AIView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var viewModel = AIViewModel()
    @State private var detail: MetricDetail?
    @State private var stockDetail: SymbolID?
    #if os(iOS)
    @Environment(\.horizontalSizeClass) private var hSize
    @Environment(\.verticalSizeClass) private var vSize
    #endif

    private var cur: String { appState.displayCurrency }

    private static func md(_ s: String) -> AttributedString {
        (try? AttributedString(markdown: s, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))) ?? AttributedString(s)
    }
    private func tone(_ score: Double) -> Color { score == 0 ? .secondary : (score >= 8 ? .green : (score >= 5 ? .yellow : .red)) }
    private func letterGrade(_ avg: Double) -> String { avg >= 8.5 ? "A" : (avg >= 7 ? "B" : (avg >= 5.5 ? "C" : "D")) }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            content
        }
        .macMinSize(width: 820, height: 560)
        .task { if viewModel.review == nil { await viewModel.load(currency: cur, accounts: appState.accountsQuery, refresh: false) } }
        .onChange(of: signature) { _, _ in Task { await viewModel.load(currency: cur, accounts: appState.accountsQuery, refresh: false) } }
        .onReceive(NotificationCenter.default.publisher(for: .refreshRequested)) { _ in
            Task { await viewModel.load(currency: cur, accounts: appState.accountsQuery, refresh: true) }
        }
        .sheet(item: $detail) { metricSheet($0) }
        .sheet(item: $stockDetail) { StockDetailView(symbol: $0.id, currency: cur) }
    }

    // MARK: - Header

    private var overall: (avg: Double, grade: String)? {
        let dims = [viewModel.review?.scorecard?.businessQuality, viewModel.review?.scorecard?.valueDiscipline, viewModel.review?.scorecard?.thesisIntegrity]
            .compactMap { $0 }.filter { $0 > 0 }
        guard !dims.isEmpty else { return nil }
        let avg = dims.reduce(0, +) / Double(dims.count)
        return (avg, letterGrade(avg))
    }

    private var header: some View {
        #if os(iOS)
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Portfolio AI Review").font(.title2.bold())
                    .foregroundStyle(LinearGradient(colors: [.purple, .pink], startPoint: .leading, endPoint: .trailing))
                Text("AI-driven insights and recommendations for your portfolio.")
                    .font(.caption).foregroundStyle(.secondary)
                if let gen = viewModel.review?.generatedAt, !gen.isEmpty {
                    Label("Generated \(gen)", systemImage: "clock").font(.caption2).foregroundStyle(.secondary)
                }
            }
            if let o = overall {
                HStack(spacing: 8) {
                    Text(o.grade).font(.system(size: 34, weight: .black)).foregroundStyle(tone(o.avg))
                    VStack(alignment: .leading) {
                        Text("Overall").font(.caption2).foregroundStyle(.secondary).textCase(.uppercase)
                        Text("\(String(format: "%.1f", o.avg))/10").font(.caption.bold()).foregroundStyle(tone(o.avg))
                    }
                }
                .padding(.horizontal, 12).padding(.vertical, 8)
                .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
                .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
            }
            Button { Task { await viewModel.load(currency: cur, accounts: appState.accountsQuery, refresh: true) } } label: {
                if viewModel.isRefreshing { HStack { ProgressView().controlSize(.small); Text("Analyzing…") } }
                else { Label("Refresh Analysis", systemImage: "arrow.clockwise") }
            }
            .buttonStyle(.bordered).disabled(viewModel.isRefreshing)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal, 20).padding(.vertical, 12)
        #else
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Portfolio AI Review").font(.title2.bold())
                    .foregroundStyle(LinearGradient(colors: [.purple, .pink], startPoint: .leading, endPoint: .trailing))
                Text("AI-driven insights and recommendations for your portfolio.")
                    .font(.caption).foregroundStyle(.secondary)
                if let gen = viewModel.review?.generatedAt, !gen.isEmpty {
                    Label("Generated \(gen)", systemImage: "clock").font(.caption2).foregroundStyle(.secondary)
                }
            }
            Spacer()
            if let o = overall {
                HStack(spacing: 8) {
                    Text(o.grade).font(.system(size: 34, weight: .black)).foregroundStyle(tone(o.avg))
                    VStack(alignment: .leading) {
                        Text("Overall").font(.caption2).foregroundStyle(.secondary).textCase(.uppercase)
                        Text("\(String(format: "%.1f", o.avg))/10").font(.caption.bold()).foregroundStyle(tone(o.avg))
                    }
                }
                .padding(.horizontal, 12).padding(.vertical, 8)
                .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
                .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
            }
            Button { Task { await viewModel.load(currency: cur, accounts: appState.accountsQuery, refresh: true) } } label: {
                if viewModel.isRefreshing { HStack { ProgressView().controlSize(.small); Text("Analyzing…") } }
                else { Label("Refresh Analysis", systemImage: "arrow.clockwise") }
            }
            .buttonStyle(.bordered).disabled(viewModel.isRefreshing)
        }
        .padding(.horizontal, 20).padding(.vertical, 12)
        #endif
    }

    // MARK: - Content

    @ViewBuilder private var content: some View {
        if viewModel.isLoading {
            ProgressView("Generating analysis…").frame(maxWidth: .infinity, maxHeight: .infinity)
        } else if let review = viewModel.review, review.scorecard != nil || review.summary != nil {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    if let w = review.warning ?? (review.error == "RateLimit" ? review.error : nil) {
                        banner(review.message ?? "AI service busy. Showing cached analysis.", warning: w)
                    }
                    scorecardGrid(review)
                    if let summary = review.summary, !summary.isEmpty { sectionCard("Executive Summary", summary, accent: .pink) }
                    if !review.optimizations.isEmpty { optimizationHub(review.optimizations) }
                    detailedAnalysis(review)
                }
                .padding(20)
            }
        } else {
            unavailable
        }
    }

    private var unavailable: some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle").font(.largeTitle).foregroundStyle(.red)
            Text("Unable to generate analysis").font(.headline)
            Text(viewModel.errorMessage ?? viewModel.review?.message ?? "Generate an AI review of your portfolio.")
                .foregroundStyle(.secondary).multilineTextAlignment(.center)
            Button { Task { await viewModel.load(currency: cur, accounts: appState.accountsQuery, refresh: true) } } label: {
                Label("Generate Analysis", systemImage: "sparkles")
            }.buttonStyle(.borderedProminent)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity).padding(40)
    }

    private func banner(_ msg: String, warning: String) -> some View {
        HStack { Image(systemName: "exclamationmark.triangle.fill"); Text(msg).font(.callout); Spacer() }
            .foregroundStyle(.yellow).padding(12)
            .background(.yellow.opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
    }

    // MARK: - Scorecard

    private func scorecardGrid(_ review: PortfolioAIReview) -> some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 180), spacing: 12)], spacing: 12) {
            scoreCard("Business Quality", "gem", review.scorecard?.businessQuality, review.analysis?.businessQuality)
            scoreCard("Value Discipline", "scalemass", review.scorecard?.valueDiscipline, review.analysis?.valueDiscipline)
            scoreCard("Thesis Integrity", "target", review.scorecard?.thesisIntegrity, review.analysis?.thesisIntegrity)
            MetricCardView(card: MetricCard(title: "Key Actions", value: "\(review.recommendations?.count ?? review.optimizations.count)", tint: .pink))
        }
    }

    private func scoreCard(_ title: String, _ icon: String, _ score: Double?, _ analysis: String?) -> some View {
        let s = score ?? 0
        return Button {
            detail = MetricDetail(title: title, score: s, content: analysis ?? "No analysis available.")
        } label: {
            VStack(alignment: .leading, spacing: 8) {
                HStack { Image(systemName: icon).foregroundStyle(tone(s)); Spacer() }
                Text(title).font(.caption).foregroundStyle(.secondary).textCase(.uppercase)
                HStack(alignment: .firstTextBaseline, spacing: 2) {
                    Text(String(format: "%.1f", s)).font(.title.bold()).foregroundStyle(tone(s))
                    Text("/10").font(.caption).foregroundStyle(.secondary)
                }
                ProgressView(value: max(0, min(s, 10)), total: 10).tint(tone(s))
            }
            .padding(14).frame(maxWidth: .infinity, alignment: .leading)
            .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
        }
        .buttonStyle(.plain)
    }

    // MARK: - Sections

    private func sectionCard(_ title: String, _ body: String, accent: Color) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.headline).foregroundStyle(accent)
            Text(Self.md(body)).font(.callout).foregroundStyle(.primary.opacity(0.9))
        }
        .padding(16).frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
    }

    private func detailedAnalysis(_ review: PortfolioAIReview) -> some View {
        VStack(spacing: 16) {
            twoColumn(
                review.analysis?.businessQuality.map { analysisCard("Business Quality", $0) },
                review.analysis?.valueDiscipline.map { analysisCard("Value Discipline", $0) }
            )
            if let thesis = review.analysis?.thesisIntegrity { analysisCard("Thesis Integrity", thesis) }
            recommendationsCard(review)
        }
    }

    @ViewBuilder private func twoColumn(_ l: (some View)?, _ r: (some View)?) -> some View {
        if let l, let r {
            #if os(iOS)
            if hSize == .compact && vSize == .regular {
                VStack(spacing: 16) { l; r }
            } else {
                HStack(alignment: .top, spacing: 16) { l; r }
            }
            #else
            HStack(alignment: .top, spacing: 16) { l; r }
            #endif
        } else if let l { l } else if let r { r }
    }

    private func analysisCard(_ title: String, _ body: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.subheadline.weight(.semibold))
            Text(Self.md(body)).font(.caption).foregroundStyle(.secondary)
        }
        .padding(16).frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
    }

    private func recommendationsCard(_ review: PortfolioAIReview) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Actionable Recommendations", systemImage: "lightbulb").font(.headline).foregroundStyle(.pink)
            if let rec = review.analysis?.actionableRecommendations, !rec.isEmpty {
                Text(Self.md(rec)).font(.callout)
            } else if let recs = review.recommendations, !recs.isEmpty {
                ForEach(Array(recs.enumerated()), id: \.offset) { _, r in
                    Label(r, systemImage: "circle.fill").font(.callout).labelStyle(BulletStyle())
                }
            } else {
                Text("No recommendations.").foregroundStyle(.secondary)
            }
        }
        .padding(16).frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
    }

    // MARK: - Optimization Hub

    private func optimizationHub(_ opts: [PortfolioAIReview.Optimization]) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("AI Optimization Hub", systemImage: "bolt.fill").font(.headline).foregroundStyle(.orange)
            #if os(iOS)
            LazyVStack(spacing: 12) {
                ForEach(opts) { opt in optimizationCard(opt) }
            }
            #else
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 300), spacing: 12)], spacing: 12) {
                ForEach(opts) { opt in optimizationCard(opt) }
            }
            #endif
            Text("Suggestions are anchored in business fundamentals and intrinsic value, not market timing. Verify against your own thesis.")
                .font(.caption2).foregroundStyle(.secondary).multilineTextAlignment(.center).frame(maxWidth: .infinity)
        }
    }

    private func optimizationCard(_ opt: PortfolioAIReview.Optimization) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top) {
                Image(systemName: optIcon(opt.type)).foregroundStyle(optTint(opt.type)).font(.title3)
                VStack(alignment: .leading, spacing: 1) {
                    Text(opt.title).font(.callout.weight(.bold))
                    if !opt.symbol.isEmpty { Text(opt.symbol).font(.caption2.weight(.bold)).foregroundStyle(.secondary) }
                }
                Spacer()
                Text("\(opt.priority) Priority").font(.caption2.weight(.bold))
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(priorityColor(opt.priority).opacity(0.2), in: Capsule())
                    .foregroundStyle(priorityColor(opt.priority))
            }
            Text(opt.description).font(.caption).foregroundStyle(.secondary).lineLimit(4)
            HStack {
                Label("\(opt.action) Recommended", systemImage: "arrow.up.right").font(.caption2.weight(.bold))
                    .foregroundStyle(.secondary)
                Spacer()
                if !opt.symbol.isEmpty && opt.symbol != "N/A" {
                    Button("Details") { stockDetail = SymbolID(id: opt.symbol) }.font(.caption2).buttonStyle(.borderless)
                }
            }
        }
        .padding(16).frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(.quaternary, lineWidth: 1))
    }

    private func optIcon(_ type: String) -> String {
        switch type {
        case "add": return "plus.circle"; case "trim": return "minus.circle"; case "exit": return "rectangle.portrait.and.arrow.right"
        case "monitor": return "eye"; case "tax_efficiency", "tax_loss_harvesting": return "receipt"
        default: return "bolt"
        }
    }
    private func optTint(_ type: String) -> Color {
        switch type {
        case "add": return .green; case "trim": return .orange; case "exit": return .red
        case "monitor": return .indigo; default: return .purple
        }
    }
    private func priorityColor(_ p: String) -> Color { p == "High" ? .red : (p == "Medium" ? .orange : .indigo) }

    // MARK: - Metric sheet

    private func metricSheet(_ m: MetricDetail) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Score Explanation").font(.caption).foregroundStyle(.secondary).textCase(.uppercase)
                    HStack(spacing: 8) {
                        Text(m.title).font(.title3.bold()).foregroundStyle(tone(m.score))
                        Text("\(String(format: "%.1f", m.score))/10").font(.headline).foregroundStyle(tone(m.score))
                    }
                }
                Spacer()
                Button { detail = nil } label: { Image(systemName: "xmark.circle.fill") }.buttonStyle(.plain).font(.title2).foregroundStyle(.secondary)
            }
            ScrollView { Text(Self.md(m.content)).font(.callout).frame(maxWidth: .infinity, alignment: .leading) }
        }
        #if os(iOS)
        .padding(24).frame(maxWidth: .infinity, maxHeight: .infinity)
        #else
        .padding(24).frame(width: 520, height: 420)
        #endif
    }

    private var signature: String {
        "\(cur)|\(appState.selectedAccounts.sorted().joined(separator: ","))"
    }
}

/// Bulleted label style for recommendation lists.
private struct BulletStyle: LabelStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Circle().fill(.pink).frame(width: 4, height: 4)
            configuration.title
        }
    }
}
