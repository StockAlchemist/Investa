import SwiftUI

struct APIKeysSettingsView: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?

    @State private var geminiApiKey = ""
    @State private var fmpApiKey = ""
    @State private var secThApiKey = ""
    @State private var botApiKey = ""
    @State private var tiingoApiKey = ""
    @State private var showApiKeys = false
    @State private var isSaving = false

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Header Note
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("External API Keys")
                            .appFont(.title3.bold())
                        Spacer()
                        Button {
                            showApiKeys.toggle()
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: showApiKeys ? "eye.slash" : "eye")
                                Text(showApiKeys ? "Hide" : "Reveal")
                            }
                            .appFont(.caption.weight(.medium))
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }

                    Text("Configure external API keys stored in server environment for market data providers and AI models.")
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 4)

                // API Key Cards
                VStack(spacing: 14) {
                    keyField(
                        title: "Gemini API Key",
                        subtitle: "Powers Gemini 2.5 Flash for AI stock analysis, portfolio synthesis, and thesis generation.",
                        placeholder: "AIzaSy...",
                        icon: "sparkles",
                        iconColor: .cyan,
                        text: $geminiApiKey
                    )

                    keyField(
                        title: "Financial Modeling Prep (FMP) API Key",
                        subtitle: "Powers financial statements, balance sheets, cash flows, DCF valuation models, and ratios.",
                        placeholder: "Enter FMP API key",
                        icon: "chart.pie.fill",
                        iconColor: .indigo,
                        text: $fmpApiKey
                    )

                    keyField(
                        title: "Thai SEC API Key",
                        subtitle: "Retrieves daily official NAV and historical price records for Thai mutual funds (SSF, RMF, ThaiESG).",
                        placeholder: "Enter SEC Thailand API key",
                        icon: "building.columns.fill",
                        iconColor: .orange,
                        text: $secThApiKey
                    )

                    keyField(
                        title: "Bank of Thailand (BOT) API Key",
                        subtitle: "Provides official Bank of Thailand daily foreign exchange conversion rates for THB pairs.",
                        placeholder: "Enter Bank of Thailand API key",
                        icon: "banknote.fill",
                        iconColor: .green,
                        text: $botApiKey
                    )

                    keyField(
                        title: "Tiingo API Key",
                        subtitle: "Used for historical stock split adjustments, dividend validation, and market data fallback.",
                        placeholder: "Enter Tiingo API key",
                        icon: "arrow.triangle.swap",
                        iconColor: .purple,
                        text: $tiingoApiKey
                    )
                }

                // Save Button
                Button {
                    saveKeys()
                } label: {
                    HStack(spacing: 8) {
                        if isSaving {
                            ProgressView().controlSize(.small)
                        } else {
                            Image(systemName: "checkmark.circle.fill")
                        }
                        Text("Save All API Keys")
                    }
                    .frame(maxWidth: .infinity)
                    .fontWeight(.bold)
                    .padding(.vertical, 8)
                }
                .buttonStyle(.borderedProminent)
                .tint(.orange)
                .disabled(isSaving)
                .padding(.top, 8)
            }
            .padding(16)
        }
        .navigationTitle("API Keys (.env)")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
        .onAppear { seed() }
        .onChange(of: settings?.geminiApiKey) { _, new in geminiApiKey = new ?? geminiApiKey }
        .onChange(of: settings?.fmpApiKey) { _, new in fmpApiKey = new ?? fmpApiKey }
        .onChange(of: settings?.secThApiKey) { _, new in secThApiKey = new ?? secThApiKey }
        .onChange(of: settings?.botApiKey) { _, new in botApiKey = new ?? botApiKey }
        .onChange(of: settings?.tiingoApiKey) { _, new in tiingoApiKey = new ?? tiingoApiKey }
    }

    private func seed() {
        geminiApiKey = settings?.geminiApiKey ?? ""
        fmpApiKey = settings?.fmpApiKey ?? ""
        secThApiKey = settings?.secThApiKey ?? ""
        botApiKey = settings?.botApiKey ?? ""
        tiingoApiKey = settings?.tiingoApiKey ?? ""
    }

    private func keyField(
        title: String,
        subtitle: String,
        placeholder: String,
        icon: String,
        iconColor: Color,
        text: Binding<String>
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                SettingsIconBadge(icon: icon, color: iconColor, size: 24, iconSize: 13)
                Text(title)
                    .appFont(.subheadline.bold())
                Spacer()
                if !text.wrappedValue.isEmpty {
                    Text("Configured")
                        .appFont(.system(size: 10, weight: .bold))
                        .foregroundStyle(.green)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.green.opacity(0.12), in: Capsule())
                }
            }

            Text(subtitle)
                .appFont(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            if showApiKeys {
                TextField(placeholder, text: text)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled()
            } else {
                SecureField(placeholder, text: text)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled()
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color.primary.opacity(0.03))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
        )
    }

    private func saveKeys() {
        isSaving = true
        Task {
            await vm.updateAPIKeys(
                gemini: geminiApiKey,
                fmp: fmpApiKey,
                secTh: secThApiKey,
                bot: botApiKey,
                tiingo: tiingoApiKey
            )
            isSaving = false
            ToastManager.shared.show(message: "API keys updated successfully", style: .success)
        }
    }
}
