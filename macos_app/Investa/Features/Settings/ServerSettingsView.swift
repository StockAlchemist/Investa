import SwiftUI

struct ServerSettingsView: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?

    var embedded: Bool = false
    @State private var serverURL = APIConfig.baseURL
    @State private var isClearingCache = false

    var body: some View {
        Group {
            if embedded {
                mainContent
            } else {
                ScrollView {
                    mainContent
                        .padding(16)
                }
                .navigationTitle("System & Server")
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
            }
        }
    }

    private var mainContent: some View {
        VStack(spacing: 20) {
            // Server Base URL Card
                VStack(alignment: .leading, spacing: 14) {
                    HStack(spacing: 8) {
                        SectionLabel(title: "Backend Base URL")
                        Spacer(minLength: 0)
                    }

                    Text("The address of your FastAPI backend instance. Change this if running against a remote LAN host or Tailscale node.")
                        .appFont(.caption)
                        .foregroundStyle(.secondary)

                    VStack(alignment: .leading, spacing: 8) {
                        TextField(APIConfig.fallbackBaseURL, text: $serverURL)
                            .textFieldStyle(.roundedBorder)
                            .autocorrectionDisabled()

                        HStack {
                            Button("Reset Default") {
                                // Must be APIConfig's own fallback: it carries the
                                // /api prefix, and a hand-written URL without it
                                // 404s every request with no way back but retyping.
                                serverURL = APIConfig.fallbackBaseURL
                                APIConfig.baseURL = serverURL
                                ToastManager.shared.show(message: "Reset backend URL to default", style: .info)
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)

                            Spacer()

                            Button("Save URL") {
                                APIConfig.baseURL = serverURL
                                ToastManager.shared.show(message: "Saved backend URL: \(serverURL)", style: .success)
                            }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.small)
                        }
                    }
                }
                .padding(18)
                .card()

                // Server Cache Management Card
                VStack(alignment: .leading, spacing: 14) {
                    HStack(spacing: 8) {
                        SectionLabel(title: "Market Data Cache")
                        Spacer(minLength: 0)
                    }

                    Text("Purge all locally cached ticker quotes, exchange rate tables, and financial statements to force fresh fetches.")
                        .appFont(.caption)
                        .foregroundStyle(.secondary)

                    Button {
                        clearCache()
                    } label: {
                        HStack(spacing: 6) {
                            if isClearingCache {
                                ProgressView().controlSize(.small)
                            } else {
                                Image(systemName: "trash.circle")
                            }
                            Text("Clear Server Cache")
                        }
                        .frame(maxWidth: .infinity)
                        .fontWeight(.semibold)
                    }
                    .buttonStyle(.bordered)
                    .tint(Color.brand)
                    .disabled(isClearingCache)
                }
                .padding(18)
                .card()
            }
        }

    private func clearCache() {
        isClearingCache = true
        Task {
            let cleared = await vm.clearCache()
            isClearingCache = false
            guard cleared else { return }
            ToastManager.shared.show(message: "Server cache cleared successfully", style: .success)
        }
    }
}
