import SwiftUI

struct MetricHelpButton: View {
    let title: String
    @State private var showing = false

    private var explanation: String? {
        let label = title.lowercased()
        if label.contains("twr") { return "Time-weighted return measures investment performance after removing the effect of deposits and withdrawals. Annualized figures express the equivalent yearly rate." }
        if label.contains("irr") || label.contains("mwr") { return "Money-weighted return (IRR) accounts for the amounts and dates of your cash flows. Annualized figures express the equivalent yearly rate." }
        if label.contains("yield") && label.contains("cost") { return "Estimated annual income divided by the original cost of the holdings. It differs from yield based on current market value." }
        if label.contains("yield") { return "Estimated annual dividend income divided by current market value. Future payments may change." }
        return nil
    }

    var body: some View {
        if let explanation {
            Button { showing.toggle() } label: { Image(systemName: "questionmark.circle").foregroundStyle(.secondary) }
                .buttonStyle(.plain)
                .accessibilityLabel("Explain \(title)")
                .popover(isPresented: $showing) { Text(explanation).padding().frame(maxWidth: 280).presentationCompactAdaptation(.popover) }
        }
    }
}
