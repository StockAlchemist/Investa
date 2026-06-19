import SwiftUI
import Charts

/// One line inside a chart tooltip card.
struct ChartTooltipRow: Identifiable {
    let id = UUID()
    let color: Color?
    let label: String
    let value: String
    init(color: Color? = nil, label: String, value: String) {
        self.color = color; self.label = label; self.value = value
    }
}

/// The contents of a hover tooltip for one x-position.
struct ChartTooltipContent {
    let title: String
    let rows: [ChartTooltipRow]
    init(title: String, rows: [ChartTooltipRow]) { self.title = title; self.rows = rows }
    init(title: String, _ rows: [ChartTooltipRow]) { self.title = title; self.rows = rows }
}

/// Floating card shown next to the hover cursor.
struct ChartTooltipCard: View {
    let content: ChartTooltipContent
    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            if !content.title.isEmpty {
                Text(content.title).font(.caption2.bold())
            }
            ForEach(content.rows) { r in
                HStack(spacing: 6) {
                    if let c = r.color {
                        RoundedRectangle(cornerRadius: 2).fill(c).frame(width: 8, height: 8)
                    }
                    Text(r.label).font(.caption2).foregroundStyle(.secondary)
                    Spacer(minLength: 12)
                    Text(r.value).font(.caption2.monospacedDigit().weight(.semibold))
                }
            }
        }
        .padding(.horizontal, 9).padding(.vertical, 7)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(.quaternary, lineWidth: 0.5))
        .shadow(color: .black.opacity(0.15), radius: 5, y: 2)
        .frame(maxWidth: 280)
    }
}

private struct ChartHoverTooltip<X: Plottable & Hashable>: ViewModifier {
    let xs: [X]
    let onTap: ((Int) -> Void)?
    let tooltip: (Int) -> ChartTooltipContent?
    @State private var selection: Int?
    @State private var cardSize: CGSize = .zero

    func body(content: Content) -> some View {
        content.chartOverlay { proxy in
            GeometryReader { geo in
                if let anchor = proxy.plotFrame {
                    let plot = geo[anchor]
                    ZStack(alignment: .topLeading) {
                        Rectangle().fill(.clear).contentShape(Rectangle())
                            .onContinuousHover { phase in
                                switch phase {
                                case .active(let pt):
                                    guard plot.contains(pt) else { selection = nil; return }
                                    selection = nearest(pt.x - plot.minX, proxy)
                                case .ended:
                                    selection = nil
                                }
                            }
                            #if os(iOS)
                            .gesture(
                                DragGesture(minimumDistance: 0)
                                    .onChanged { value in
                                        guard plot.contains(value.location) else { selection = nil; return }
                                        selection = nearest(value.location.x - plot.minX, proxy)
                                    }
                                    .onEnded { value in
                                        selection = nil
                                        let d = hypot(value.translation.width, value.translation.height)
                                        if d < 10, let onTap, plot.contains(value.location),
                                           let i = nearest(value.location.x - plot.minX, proxy) {
                                            onTap(i)
                                        }
                                    }
                            )
                            #else
                            .gesture(SpatialTapGesture().onEnded { value in
                                guard let onTap, plot.contains(value.location),
                                      let i = nearest(value.location.x - plot.minX, proxy) else { return }
                                onTap(i)
                            })
                            #endif
                        if let sel = selection, xs.indices.contains(sel),
                           let c = tooltip(sel), let px = proxy.position(forX: xs[sel]) {
                            let lineX = plot.minX + px
                            Rectangle().fill(Color.secondary.opacity(0.35))
                                .frame(width: 1, height: plot.height)
                                .position(x: lineX, y: plot.minY + plot.height / 2)
                                .allowsHitTesting(false)
                            let half = cardSize.width / 2 + 8
                            let cx = min(max(lineX, plot.minX + half), plot.maxX - half)
                            ChartTooltipCard(content: c)
                                .background(GeometryReader { g in
                                    Color.clear
                                        .onAppear { cardSize = g.size }
                                        .onChange(of: g.size) { _, s in cardSize = s }
                                })
                                .position(x: cx.isFinite ? cx : lineX,
                                          y: plot.minY + cardSize.height / 2 + 6)
                                .allowsHitTesting(false)
                        }
                    }
                }
            }
        }
    }

    private func nearest(_ localX: CGFloat, _ proxy: ChartProxy) -> Int? {
        var best: Int?
        var bestD = CGFloat.greatestFiniteMagnitude
        for (i, xv) in xs.enumerated() {
            if let px = proxy.position(forX: xv) {
                let d = abs(px - localX)
                if d < bestD { bestD = d; best = i }
            }
        }
        return best
    }
}

extension View {
    /// Adds a hover cursor + floating tooltip to a line/bar/area chart, mirroring
    /// the web app's chart tooltips. `xs` is the chart's distinct x-values in plot
    /// order; `tooltip(i)` builds the card for the hovered x. Pass `onTap` to also
    /// handle clicks on the nearest x (e.g. year filtering).
    func chartHoverTooltip<X: Plottable & Hashable>(
        _ xs: [X],
        onTap: ((Int) -> Void)? = nil,
        tooltip: @escaping (Int) -> ChartTooltipContent?
    ) -> some View {
        modifier(ChartHoverTooltip(xs: xs, onTap: onTap, tooltip: tooltip))
    }
}
