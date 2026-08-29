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
                        Image(systemName: "network")
                            .foregroundStyle(Color.blue)
                            .appFont(.title3)
                        Text("Backend Base URL")
                            .appFont(.headline.bold())
                        Spacer()
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
                .background(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(Color.primary.opacity(0.03))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
                )

                // Server Cache Management Card
                VStack(alignment: .leading, spacing: 14) {
                    HStack(spacing: 8) {
                        Image(systemName: "internaldrive.fill")
                            .foregroundStyle(Color.orange)
                            .appFont(.title3)
                        Text("Market Data Cache")
                            .appFont(.headline.bold())
                        Spacer()
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
                    .tint(.orange)
                    .disabled(isClearingCache)
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
