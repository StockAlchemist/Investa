import SwiftUI

struct BenchmarksSettingsView: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?
    @EnvironmentObject private var appState: AppState

    var embedded: Bool = false
    @State private var customTicker = ""

    private let presetBenchmarks = [
        "S&P 500", "NASDAQ", "Dow Jones", "Russell 2000",
        "Total US Market (VTI)", "All-World (VT)", "Total International (VXUS)", "Emerging Markets (VWO)",
        "Europe (VGK)", "Japan (EWJ)", "US Total Bond (BND)", "20+ Year Treasury (TLT)",
        "Gold (GLD)", "Bitcoin (BTC-USD)", "US Growth (VUG)", "US Value (VTV)", "US Dividend (SCHD)",
    ]

    var body: some View {
        Group {
            if embedded {
                mainContent
            } else {
                ScrollView {
                    mainContent
                        .padding(16)
                }
                .navigationTitle("Benchmarks")
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
            }
        }
    }

    private var mainContent: some View {
        VStack(spacing: 20) {
            // Intro Note
            Text("Select market indices and custom benchmark tickers to compare your portfolio returns and performance charts against.")
                .appFont(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 4)

            // Presets Section
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Popular Index Benchmarks")
                        .appFont(.headline.bold())
                    Spacer()
                }

                #if os(macOS)
                let cols = [GridItem(.adaptive(minimum: 180), spacing: 8)]
                #else
                let cols = [GridItem(.adaptive(minimum: 150), spacing: 8)]
                #endif

                LazyVGrid(columns: cols, alignment: .leading, spacing: 8) {
                    ForEach(presetBenchmarks, id: \.self) { b in
                        let on = appState.benchmarks.contains(b)
                        Button {
                            toggleBenchmark(b)
                        } label: {
                            HStack(spacing: 8) {
                                Image(systemName: on ? "checkmark.circle.fill" : "circle")
                                    .foregroundStyle(on ? Color.purple : .secondary.opacity(0.6))
                                    .font(.system(size: 16))

                                Text(b)
                                    .appFont(.caption.weight(.medium))
                                    .foregroundStyle(on ? .primary : .secondary)

                                Spacer()
                            }
                            .padding(.horizontal, 10)
                            .padding(.vertical, 8)
                            .background(on ? Color.purple.opacity(0.12) : Color.primary.opacity(0.04))
                            .cornerRadius(10)
                            .overlay(
                                RoundedRectangle(cornerRadius: 10)
                                    .stroke(on ? Color.purple.opacity(0.35) : Color.clear, lineWidth: 1)
                            )
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .padding(18)
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(Color.primary.opacity(0.03))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
            )

                // Custom Tickers Section
                VStack(alignment: .leading, spacing: 14) {
                    HStack {
                        Text("Custom Benchmark Tickers")
                            .appFont(.headline.bold())
                        Spacer()
                    }

                    HStack(spacing: 10) {
                        TextField("e.g. AAPL, QQQ, NVDA", text: $customTicker)
                            .textFieldStyle(.roundedBorder)
                            .uppercaseAutoCapitalization()
                            .autocorrectionDisabled()

                        Button {
                            addCustomBenchmark()
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: "plus")
                                Text("Add")
                            }
                            .fontWeight(.semibold)
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.purple)
                        .disabled(customTicker.trimmingCharacters(in: .whitespaces).isEmpty)
                    }

                    let customList = appState.benchmarks.filter { !presetBenchmarks.contains($0) }
                    if customList.isEmpty {
                        Text("No custom benchmark tickers added.")
                            .appFont(.caption)
                            .foregroundStyle(.secondary)
                            .padding(.top, 4)
                    } else {
                        FlowChipsRemovable(items: customList, color: .purple) { sym in
                            toggleBenchmark(sym)
                        }
                        .padding(.top, 4)
                    }
                }
                .padding(18)
                .background(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(Color.primary.opacity(0.03))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
                )
        }
    }

    private func toggleBenchmark(_ b: String) {
        if appState.benchmarks.contains(b) {
            appState.setBenchmarks(appState.benchmarks.filter { $0 != b })
            ToastManager.shared.show(message: "Removed benchmark \(b)", style: .info)
        } else {
            appState.setBenchmarks(appState.benchmarks + [b])
            ToastManager.shared.show(message: "Added benchmark \(b)", style: .success)
        }
    }

    private func addCustomBenchmark() {
        let t = customTicker.trimmingCharacters(in: .whitespaces).uppercased()
        guard !t.isEmpty, !appState.benchmarks.contains(t) else { return }
        customTicker = ""
        appState.setBenchmarks(appState.benchmarks + [t])
        ToastManager.shared.show(message: "Added custom benchmark \(t)", style: .success)
    }
}
