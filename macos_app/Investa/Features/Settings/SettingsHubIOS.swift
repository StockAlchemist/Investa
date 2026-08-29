import SwiftUI

/// Root Settings Hub for iOS (iPhone & compact views) designed to follow Apple HIG inset-grouped layout.
struct SettingsHubIOS: View {
    @ObservedObject var viewModel: SettingsViewModel
    @EnvironmentObject private var appState: AppState
    @EnvironmentObject private var auth: AuthViewModel

    private var groupsCount: Int {
        viewModel.settings?.accountGroups?.count ?? 0
    }

    private var mappingsCount: Int {
        viewModel.settings?.userSymbolMap?.count ?? 0
    }

    private var excludedCount: Int {
        viewModel.settings?.userExcludedSymbols?.count ?? 0
    }

    private var overridesCount: Int {
        viewModel.settings?.manualOverrides?.count ?? 0
    }

    private var benchmarksCount: Int {
        appState.benchmarks.count
    }

    var body: some View {
        List {
            // Profile Hero Section
            Section {
                NavigationLink {
                    ProfileSecuritySettingsView(vm: viewModel)
                        .environmentObject(auth)
                } label: {
                    profileHeroRow
                }
            }

            // Section 1: Portfolio & Accounts
            Section("Portfolio & Accounts") {
                SettingsNavRow(
                    icon: "person.2.fill",
                    iconColor: .indigo,
                    title: "Custom Account Groups",
                    subtitle: "Group accounts for aggregate filtering",
                    badge: groupsCount > 0 ? "\(groupsCount)" : nil
                ) {
                    AccountGroupManagerView(
                        vm: viewModel,
                        settings: viewModel.settings,
                        availableAccounts: appState.allAccounts,
                        appState: appState
                    )
                }

                SettingsNavRow(
                    icon: "slider.horizontal.3",
                    iconColor: .purple,
                    title: "Account Preferences",
                    subtitle: "Currencies, cash modes & closure dates"
                ) {
                    AccountPreferencesView(
                        vm: viewModel,
                        settings: viewModel.settings,
                        accounts: appState.allAccounts,
                        appState: appState
                    )
                }

                SettingsNavRow(
                    icon: "dollarsign.circle.fill",
                    iconColor: .orange,
                    title: "Currency Management",
                    subtitle: "Add/remove available currencies"
                ) {
                    CurrencyManagementView(
                        vm: viewModel,
                        settings: viewModel.settings
                    )
                }

                SettingsNavRow(
                    icon: "percent",
                    iconColor: .teal,
                    title: "Cash Yield Assumptions",
                    subtitle: "Interest rates & exempt thresholds"
                ) {
                    CashYieldSettingsView(
                        vm: viewModel,
                        settings: viewModel.settings,
                        accounts: appState.allAccounts,
                        appState: appState
                    )
                }
            }

            // Section 2: Data & Mapping
            Section("Data & Ticker Mapping") {
                SettingsNavRow(
                    icon: "arrow.left.arrow.right",
                    iconColor: .blue,
                    title: "Symbol Mappings",
                    subtitle: "Resolve custom tickers to Yahoo Finance",
                    badge: mappingsCount > 0 ? "\(mappingsCount)" : nil
                ) {
                    SymbolMappingsView(
                        vm: viewModel,
                        settings: viewModel.settings
                    )
                }

                SettingsNavRow(
                    icon: "xmark.circle.fill",
                    iconColor: .red,
                    title: "Excluded Symbols",
                    subtitle: "Skip symbols from portfolio calculation",
                    badge: excludedCount > 0 ? "\(excludedCount)" : nil
                ) {
                    ExcludedSymbolsView(
                        vm: viewModel,
                        settings: viewModel.settings
                    )
                }

                SettingsNavRow(
                    icon: "pencil.and.ruler.fill",
                    iconColor: .green,
                    title: "Manual Overrides",
                    subtitle: "Manual prices & metadata overrides",
                    badge: overridesCount > 0 ? "\(overridesCount)" : nil
                ) {
                    OverridesListView(
                        vm: viewModel,
                        settings: viewModel.settings
                    )
                }
            }

            // Section 3: Integrations & Performance
            Section("Integrations & Benchmarks") {
                SettingsNavRow(
                    icon: "chart.line.uptrend.xyaxis",
                    iconColor: .purple,
                    title: "Performance Benchmarks",
                    subtitle: "Market indices & custom comparison tickers",
                    badge: benchmarksCount > 0 ? "\(benchmarksCount)" : nil
                ) {
                    BenchmarksSettingsView(
                        vm: viewModel,
                        settings: viewModel.settings
                    )
                    .environmentObject(appState)
                }

                SettingsNavRow(
                    icon: "arrow.triangle.2.circlepath",
                    iconColor: .blue,
                    title: "Interactive Brokers & Webhook",
                    subtitle: "IBKR Flex query sync & data refresh"
                ) {
                    IntegrationsSettingsView(
                        vm: viewModel,
                        settings: viewModel.settings
                    )
                }
            }

            // Section 4: Developer & System
            Section("Developer & System") {
                SettingsNavRow(
                    icon: "key.fill",
                    iconColor: .orange,
                    title: "API Keys (.env)",
                    subtitle: "Gemini, FMP, SEC TH, BOT, Tiingo"
                ) {
                    APIKeysSettingsView(
                        vm: viewModel,
                        settings: viewModel.settings
                    )
                }

                SettingsNavRow(
                    icon: "server.rack",
                    iconColor: .gray,
                    title: "System & Server",
                    subtitle: "Backend URL & cache purge"
                ) {
                    ServerSettingsView(
                        vm: viewModel,
                        settings: viewModel.settings
                    )
                }
            }

            // Section 5: Account & Security
            Section("Account & Security") {
                SettingsNavRow(
                    icon: "person.crop.circle.fill",
                    iconColor: .cyan,
                    title: "Profile & Password",
                    subtitle: "Display name and password change"
                ) {
                    ProfileSecuritySettingsView(vm: viewModel)
                        .environmentObject(auth)
                }

                Button(role: .destructive) {
                    auth.logout()
                } label: {
                    HStack(spacing: 12) {
                        SettingsIconBadge(icon: "rectangle.portrait.and.arrow.right", color: .red)
                        Text("Sign Out")
                            .appFont(.body)
                            .foregroundStyle(.red)
                    }
                    .padding(.vertical, 2)
                }
            }

            // Footer
            Section {
                VStack(spacing: 4) {
                    Text("Investa • Version 0.1.0")
                        .appFont(.caption2.bold())
                        .foregroundStyle(.secondary)
                    Text("FastAPI + Next.js + SwiftUI")
                        .appFont(.system(size: 10))
                        .foregroundStyle(.secondary.opacity(0.7))
                }
                .frame(maxWidth: .infinity)
                .listRowBackground(Color.clear)
                .padding(.vertical, 8)
            }
        }
        #if os(iOS)
        .listStyle(.insetGrouped)
        .navigationBarTitleDisplayMode(.large)
        #else
        .listStyle(.sidebar)
        #endif
        .navigationTitle("Settings")
    }

    private var profileHeroRow: some View {
        HStack(spacing: 14) {
            ZStack {
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [.cyan, .blue],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 52, height: 52)
                    .shadow(color: .cyan.opacity(0.35), radius: 6, x: 0, y: 3)

                Text(avatarInitial)
                    .appFont(.title3.bold())
                    .foregroundStyle(.white)
            }

            VStack(alignment: .leading, spacing: 3) {
                Text(auth.currentUser?.displayName ?? "Investa User")
                    .appFont(.headline.bold())
                    .foregroundStyle(.primary)

                Text("@\(auth.currentUser?.username ?? "user")")
                    .appFont(.subheadline)
                    .foregroundStyle(.secondary)
            }

            Spacer()
        }
        .padding(.vertical, 6)
    }

    private var avatarInitial: String {
        let name = auth.currentUser?.displayName ?? auth.currentUser?.username ?? "U"
        return String(name.prefix(1)).uppercased()
    }
}
