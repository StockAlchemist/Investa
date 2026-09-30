import SwiftUI

struct ServerSettingsView: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?

    var embedded: Bool = false
    @ObservedObject private var store = ServerStore.shared
    @State private var serverURL = APIConfig.baseURL
    @State private var serverName = ""
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
            // Backend server card: saved servers to switch between, then the
            // address being edited.
                VStack(alignment: .leading, spacing: 14) {
                    HStack(spacing: 8) {
                        SectionLabel(title: "Backend Server")
                        Spacer(minLength: 0)
                    }

                    Text("The address of your FastAPI backend. Save the ones you use — home LAN, Tailscale — and switch between them here or from the server menu in the top bar.")
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)

                    if !store.servers.isEmpty { savedServerList }

                    VStack(alignment: .leading, spacing: 8) {
                        TextField("Name (optional)", text: $serverName)
                            .textFieldStyle(.roundedBorder)
                            .autocorrectionDisabled()
                        TextField(APIConfig.fallbackBaseURL, text: $serverURL)
                            .textFieldStyle(.roundedBorder)
                            .autocorrectionDisabled()
                            #if os(iOS)
                            .keyboardType(.URL)
                            .textInputAutocapitalization(.never)
                            #endif

                        // Three buttons beside each other outrun a phone at a
                        // large type size; Reset drops to its own line there.
                        ViewThatFits(in: .horizontal) {
                            HStack {
                                resetButton
                                Spacer(minLength: 8)
                                saveButton
                                connectButton
                            }
                            VStack(alignment: .leading, spacing: 8) {
                                HStack {
                                    saveButton
                                    Spacer(minLength: 8)
                                    connectButton
                                }
                                resetButton
                            }
                        }
                        .controlSize(.small)
                    }
                }
                .padding(18)
                .card()
                // The control bar's quick switcher moves the active server
                // under this card; keep the field showing the one in use.
                .onChange(of: store.activeURL) { _, url in serverURL = url }

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

    private var savedServerList: some View {
        VStack(spacing: 0) {
            ForEach(Array(store.servers.enumerated()), id: \.element.id) { index, server in
                if index > 0 { Rectangle().fill(Color.line).frame(height: 1) }
                savedServerRow(server)
            }
        }
        .background(Color.inset, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    private func savedServerRow(_ server: SavedServer) -> some View {
        let active = store.isActive(server)
        return HStack(spacing: 10) {
            Button { connect(to: server) } label: {
                HStack(spacing: 10) {
                    Image(systemName: active ? "checkmark.circle.fill" : "circle")
                        .foregroundStyle(active ? Color.brand : Color.ink3)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(server.name)
                            .appFont(.callout.weight(.semibold))
                            .foregroundStyle(.primary)
                        Text(server.url)
                            .appFont(.caption).monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                    Spacer(minLength: 0)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityAddTraits(active ? .isSelected : [])

            Button {
                store.remove(server)
            } label: {
                Image(systemName: "trash")
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .accessibilityLabel("Remove \(server.name)")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
    }

    private var resetButton: some View {
        Button("Reset Default") {
            // Must be APIConfig's own fallback: it carries the /api prefix,
            // and a hand-written URL without it 404s every request with no
            // way back but retyping.
            serverName = ""
            serverURL = APIConfig.fallbackBaseURL
            store.switchTo(serverURL)
            ToastManager.shared.show(message: "Reset backend URL to default", style: .info)
        }
        .buttonStyle(.bordered)
    }

    private var saveButton: some View {
        Button("Save to List") {
            guard let saved = store.add(name: serverName, url: serverURL) else { return }
            serverName = ""
            ToastManager.shared.show(message: "Saved server \(saved.name)", style: .success)
        }
        .buttonStyle(.bordered)
        .disabled(serverURL.trimmingCharacters(in: .whitespaces).isEmpty)
    }

    private var connectButton: some View {
        Button("Connect") {
            store.switchTo(serverURL)
            let name = store.activeServer?.name ?? store.activeURL
            ToastManager.shared.show(message: "Connected to \(name)", style: .success)
        }
        .buttonStyle(.borderedProminent)
        .disabled(serverURL.trimmingCharacters(in: .whitespaces).isEmpty)
    }

    private func connect(to server: SavedServer) {
        guard !store.isActive(server) else { return }
        store.switchTo(server.url)
        ToastManager.shared.show(message: "Switched to \(server.name)", style: .success)
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
