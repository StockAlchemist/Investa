import SwiftUI

struct APIKeysSettingsView: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?

    var embedded: Bool = false
    @State private var geminiApiKey = ""
    @State private var fmpApiKey = ""
    @State private var secThApiKey = ""
    @State private var botApiKey = ""
    @State private var tiingoApiKey = ""
    @State private var showApiKeys = false
    @State private var isSaving = false
    /// The last server-sent preview applied to each field, keyed by wire name.
    /// A field whose text differs from its baseline is one the user typed into:
    /// only those are posted. The server sends masked previews rather than the
    /// real keys and reads "" as "clear", so posting an untouched field would
    /// destroy the stored key.
    @State private var seededValues: [String: String] = [:]

    var body: some View {
        Group {
            if embedded {
                mainContent
            } else {
                ScrollView {
                    mainContent
                        .padding(16)
                }
                .navigationTitle("API Keys (.env)")
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
            }
        }
        .onAppear { seed() }
        // Reseed from the server only where the user is not mid-edit: saving
        // an unrelated section reloads settings, and that would otherwise
        // discard a pasted-but-unsaved key.
        .onChange(of: settings?.geminiApiKey) { _, new in reseed(Field.gemini, &geminiApiKey, new) }
        .onChange(of: settings?.fmpApiKey) { _, new in reseed(Field.fmp, &fmpApiKey, new) }
        .onChange(of: settings?.secThApiKey) { _, new in reseed(Field.secTh, &secThApiKey, new) }
        .onChange(of: settings?.botApiKey) { _, new in reseed(Field.bot, &botApiKey, new) }
        .onChange(of: settings?.tiingoApiKey) { _, new in reseed(Field.tiingo, &tiingoApiKey, new) }
    }

    private var mainContent: some View {
        VStack(spacing: 20) {
            // Header Note
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        SectionLabel(title: "External API Keys")
                        Spacer(minLength: 8)
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
                        text: $geminiApiKey
                    )

                    keyField(
                        title: "Financial Modeling Prep (FMP) API Key",
                        subtitle: "Powers financial statements, balance sheets, cash flows, DCF valuation models, and ratios.",
                        placeholder: "Enter FMP API key",
                        icon: "chart.pie.fill",
                        text: $fmpApiKey
                    )

                    keyField(
                        title: "Thai SEC API Key",
                        subtitle: "Retrieves daily official NAV and historical price records for Thai mutual funds (SSF, RMF, ThaiESG).",
                        placeholder: "Enter SEC Thailand API key",
                        icon: "building.columns.fill",
                        text: $secThApiKey
                    )

                    keyField(
                        title: "Bank of Thailand (BOT) API Key",
                        subtitle: "Provides official Bank of Thailand daily foreign exchange conversion rates for THB pairs.",
                        placeholder: "Enter Bank of Thailand API key",
                        icon: "banknote.fill",
                        text: $botApiKey
                    )

                    keyField(
                        title: "Tiingo API Key",
                        subtitle: "Used for historical stock split adjustments, dividend validation, and market data fallback.",
                        placeholder: "Enter Tiingo API key",
                        icon: "arrow.triangle.swap",
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
                .tint(Color.brand)
                .disabled(isSaving)
                .padding(.top, 8)
        }
    }

    /// Wire field names, matching the backend's `SettingsUpdate` model.
    private enum Field {
        static let gemini = "gemini_api_key"
        static let fmp = "fmp_api_key"
        static let secTh = "sec_th_api_key"
        static let bot = "bot_api_key"
        static let tiingo = "tiingo_api_key"
    }

    private func seed() {
        setSeed(Field.gemini, &geminiApiKey, settings?.geminiApiKey)
        setSeed(Field.fmp, &fmpApiKey, settings?.fmpApiKey)
        setSeed(Field.secTh, &secThApiKey, settings?.secThApiKey)
        setSeed(Field.bot, &botApiKey, settings?.botApiKey)
        setSeed(Field.tiingo, &tiingoApiKey, settings?.tiingoApiKey)
    }

    /// Adopt a server value and record it as the field's baseline.
    private func setSeed(_ field: String, _ binding: inout String, _ value: String?) {
        binding = value ?? ""
        seededValues[field] = binding
    }

    /// Adopt a refreshed server value only where the user is not mid-edit.
    private func reseed(_ field: String, _ binding: inout String, _ value: String?) {
        guard !isEdited(field, binding) else { return }
        setSeed(field, &binding, value)
    }

    private func isEdited(_ field: String, _ current: String) -> Bool {
        current != (seededValues[field] ?? "")
    }

    private func keyField(
        title: String,
        subtitle: String,
        placeholder: String,
        icon: String,
        text: Binding<String>
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                SettingsIcon(icon: icon, size: 20)
                Text(title)
                    .appFont(.subheadline.bold())
                Spacer()
                if !text.wrappedValue.isEmpty {
                    Text("Configured")
                        .appFont(.system(size: 10, weight: .bold))
                        .foregroundStyle(Color.up)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.up.opacity(0.12), in: Capsule())
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
        .card(.inset)
    }

    private func saveKeys() {
        let values = [
            Field.gemini: geminiApiKey,
            Field.fmp: fmpApiKey,
            Field.secTh: secThApiKey,
            Field.bot: botApiKey,
            Field.tiingo: tiingoApiKey,
        ].filter { isEdited($0.key, $0.value) }

        guard !values.isEmpty else {
            ToastManager.shared.show(message: "No API key changes to save", style: .info)
            return
        }

        isSaving = true
        Task {
            let saved = await vm.updateAPIKeys(values)
            isSaving = false
            guard saved else { return }
            // updateAPIKeys reloads settings before returning, so this picks up
            // the freshly masked previews and resets every baseline.
            seed()
            ToastManager.shared.show(message: "API keys updated successfully", style: .success)
        }
    }
}
