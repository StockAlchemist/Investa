import SwiftUI

@MainActor
final class SettingsViewModel: ObservableObject {
    @Published var settings: AppSettings?
    @Published var overridePrices: [String: Double] = [:]
    @Published var isLoading = false
    @Published var statusMessage: String?

    private let api: APIClient
    init(api: APIClient = .shared) { self.api = api }

    func load() async {
        isLoading = true
        defer { isLoading = false }
        do {
            let s: AppSettings = try await api.get("/settings")
            settings = s
            overridePrices = s.manualOverridePrices
        } catch { statusMessage = (error as? APIError)?.errorDescription ?? error.localizedDescription }
    }

    /// Persist a subset of settings via `POST /api/settings/update`.
    func update(displayCurrency: String?, showClosed: Bool?, benchmarks: [String]?) async {
        struct Update: Encodable {
            let display_currency: String?
            let show_closed: Bool?
            let benchmarks: [String]?
        }
        do {
            let _: StatusResponse = try await api.send(
                method: "POST", path: "/settings/update",
                body: Update(display_currency: displayCurrency, show_closed: showClosed, benchmarks: benchmarks)
            )
            statusMessage = "Saved."
        } catch { statusMessage = (error as? APIError)?.errorDescription ?? error.localizedDescription }
    }

    func setOverride(symbol: String, price: Double?) async {
        struct Body: Encodable { let symbol: String; let price: Double? }
        do {
            let _: StatusResponse = try await api.send(
                method: "POST", path: "/settings/manual_overrides",
                body: Body(symbol: symbol, price: price)
            )
            if let price { overridePrices[symbol] = price } else { overridePrices.removeValue(forKey: symbol) }
        } catch { statusMessage = (error as? APIError)?.errorDescription ?? error.localizedDescription }
    }

    func clearCache() async {
        do {
            let _: StatusResponse = try await api.send(method: "POST", path: "/clear_cache")
            statusMessage = "Cache cleared."
        } catch { statusMessage = (error as? APIError)?.errorDescription ?? error.localizedDescription }
    }
}

struct SettingsView: View {
    @EnvironmentObject private var appState: AppState
    @EnvironmentObject private var auth: AuthViewModel
    @StateObject private var viewModel = SettingsViewModel()

    @State private var benchmarksText = ""
    @State private var newOverrideSymbol = ""
    @State private var newOverridePrice = ""
    @State private var serverURL = APIConfig.baseURL
    @State private var showingPassword = false

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Settings").font(.title2.bold())
                if viewModel.isLoading { ProgressView().controlSize(.small) }
                Spacer()
                if let msg = viewModel.statusMessage {
                    Text(msg).font(.caption).foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, 20).padding(.vertical, 12)
            Divider()
            Form {
                generalSection
                benchmarksSection
                manualOverridesSection
                serverSection
                accountSection
            }
            .formStyle(.grouped)
        }
        .frame(minWidth: 700, minHeight: 560)
        .task {
            await viewModel.load()
            benchmarksText = (viewModel.settings?.benchmarks ?? []).joined(separator: ", ")
        }
        .sheet(isPresented: $showingPassword) { ChangePasswordView().environmentObject(auth) }
    }

    private var generalSection: some View {
        Section("Display") {
            Picker("Display Currency", selection: $appState.displayCurrency) {
                ForEach(appState.availableCurrencies, id: \.self) { Text($0).tag($0) }
            }
            .onChange(of: appState.displayCurrency) { _, new in
                Task { await viewModel.update(displayCurrency: new, showClosed: nil, benchmarks: nil) }
            }
            Toggle("Include closed accounts", isOn: $appState.showClosed)
                .onChange(of: appState.showClosed) { _, new in
                    Task { await viewModel.update(displayCurrency: nil, showClosed: new, benchmarks: nil) }
                }
        }
    }

    private var benchmarksSection: some View {
        Section("Benchmarks") {
            TextField("Comma-separated (e.g. SPY, QQQ)", text: $benchmarksText)
            Button("Save Benchmarks") {
                let list = benchmarksText.split(separator: ",")
                    .map { $0.trimmingCharacters(in: .whitespaces).uppercased() }.filter { !$0.isEmpty }
                Task { await viewModel.update(displayCurrency: nil, showClosed: nil, benchmarks: list) }
            }
        }
    }

    private var manualOverridesSection: some View {
        Section("Manual Price Overrides") {
            HStack {
                TextField("Symbol", text: $newOverrideSymbol).frame(width: 100)
                TextField("Price", text: $newOverridePrice).frame(width: 100)
                Button("Add") {
                    let sym = newOverrideSymbol.trimmingCharacters(in: .whitespaces).uppercased()
                    guard !sym.isEmpty, let price = Double(newOverridePrice) else { return }
                    newOverrideSymbol = ""; newOverridePrice = ""
                    Task { await viewModel.setOverride(symbol: sym, price: price) }
                }
            }
            ForEach(viewModel.overridePrices.sorted(by: { $0.key < $1.key }), id: \.key) { sym, price in
                HStack {
                    Text(sym).fontWeight(.medium)
                    Spacer()
                    Text(Fmt.number(price))
                    Button(role: .destructive) {
                        Task { await viewModel.setOverride(symbol: sym, price: nil) }
                    } label: { Image(systemName: "trash") }
                        .buttonStyle(.borderless)
                }
            }
        }
    }

    private var serverSection: some View {
        Section("Backend Server") {
            TextField("Base URL", text: $serverURL)
            Button("Save Server URL") { APIConfig.baseURL = serverURL }
            Button("Clear Server Cache") { Task { await viewModel.clearCache() } }
        }
    }

    private var accountSection: some View {
        Section("Account") {
            if let user = auth.currentUser {
                LabeledContent("Signed in as", value: user.displayName)
            }
            Button("Change Password…") { showingPassword = true }
            Button("Log Out", role: .destructive) { auth.logout() }
        }
    }
}
