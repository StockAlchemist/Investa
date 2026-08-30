import SwiftUI

/// Settings on iPhone (and any compact width).
///
/// The categories are grouped cards of touch-height rows rather than an
/// inset-grouped `List`: same `.card()` chrome, same `SectionLabel` heads and
/// same indigo accent as every other tab, so Settings stops being the one
/// screen drawn in the system's grey grouped style with five coloured tiles.
struct SettingsHubIOS: View {
    @ObservedObject var viewModel: SettingsViewModel
    @EnvironmentObject private var appState: AppState
    @EnvironmentObject private var auth: AuthViewModel

    private var groupsCount: Int { viewModel.settings?.accountGroups?.count ?? 0 }
    private var currenciesCount: Int { viewModel.settings?.availableCurrencies?.count ?? 0 }
    private var mappingsCount: Int { viewModel.settings?.userSymbolMap?.count ?? 0 }
    private var excludedCount: Int { viewModel.settings?.userExcludedSymbols?.count ?? 0 }
    private var overridesCount: Int { viewModel.settings?.manualOverrides?.count ?? 0 }
    private var benchmarksCount: Int { appState.benchmarks.count }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                accountsSection
                symbolsSection
                overridesSection
                advancedSection
                appearanceSection
                profileSection
                footer
            }
            .padding(20)
        }
        .navigationTitle("Settings")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.large)
        #endif
    }

    // MARK: - Sections

    private var accountsSection: some View {
        section("Accounts") {
            SettingsRowGroup {
                SettingsNavRow(
                    icon: "person.2",
                    title: "Account Groups",
                    subtitle: "Aggregate filtering",
                    count: groupsCount
                ) {
                    AccountGroupManagerView(
                        vm: viewModel,
                        settings: viewModel.settings,
                        availableAccounts: appState.allAccounts,
                        appState: appState
                    )
                }
                SettingsRowDivider()
                SettingsNavRow(
                    icon: "slider.horizontal.3",
                    title: "Account Preferences",
                    subtitle: "Currency, cash mode, closure"
                ) {
                    AccountPreferencesView(
                        vm: viewModel,
                        settings: viewModel.settings,
                        accounts: appState.allAccounts,
                        appState: appState
                    )
                }
                SettingsRowDivider()
                SettingsNavRow(
                    icon: "dollarsign.circle",
                    title: "Currencies",
                    subtitle: "Available for manual accounts",
                    count: currenciesCount
                ) {
                    CurrencyManagementView(vm: viewModel, settings: viewModel.settings)
                }
                SettingsRowDivider()
                SettingsNavRow(
                    icon: "percent",
                    title: "Cash Yield",
                    subtitle: "Rates & exempt thresholds"
                ) {
                    CashYieldSettingsView(
                        vm: viewModel,
                        settings: viewModel.settings,
                        accounts: appState.allAccounts,
                        appState: appState
                    )
                }
            }
        }
    }

    private var symbolsSection: some View {
        section("Symbols") {
            SettingsRowGroup {
                SettingsNavRow(
                    icon: "arrow.left.arrow.right",
                    title: "Symbol Mappings",
                    count: mappingsCount
                ) {
                    SymbolMappingsView(vm: viewModel, settings: viewModel.settings)
                }
                SettingsRowDivider()
                SettingsNavRow(
                    icon: "xmark.circle",
                    title: "Excluded Symbols",
                    count: excludedCount
                ) {
                    ExcludedSymbolsView(vm: viewModel, settings: viewModel.settings)
                }
            }
        }
    }

    private var overridesSection: some View {
        section("Overrides") {
            SettingsRowGroup {
                SettingsNavRow(
                    icon: "pencil.and.ruler",
                    title: "Manual Overrides",
                    subtitle: "Prices & metadata",
                    count: overridesCount
                ) {
                    OverridesListView(vm: viewModel, settings: viewModel.settings)
                }
            }
        }
    }

    private var advancedSection: some View {
        section("Advanced") {
            SettingsRowGroup {
                SettingsNavRow(
                    icon: "chart.line.uptrend.xyaxis",
                    title: "Benchmarks",
                    count: benchmarksCount
                ) {
                    BenchmarksSettingsView(vm: viewModel, settings: viewModel.settings)
                        .environmentObject(appState)
                }
                SettingsRowDivider()
                SettingsNavRow(
                    icon: "arrow.triangle.2.circlepath",
                    title: "Interactive Brokers",
                    subtitle: "Flex query sync & webhook"
                ) {
                    IntegrationsSettingsView(vm: viewModel, settings: viewModel.settings)
                }
                SettingsRowDivider()
                SettingsNavRow(
                    icon: "key",
                    title: "API Keys",
                    subtitle: "Gemini, FMP, SEC TH, BOT, Tiingo"
                ) {
                    APIKeysSettingsView(vm: viewModel, settings: viewModel.settings)
                }
                SettingsRowDivider()
                SettingsNavRow(
                    icon: "server.rack",
                    title: "System & Server",
                    subtitle: "Backend URL & cache purge"
                ) {
                    ServerSettingsView(vm: viewModel, settings: viewModel.settings)
                }
            }
        }
    }

    private var appearanceSection: some View {
        section("Appearance") {
            SettingsRowGroup {
                SettingsNavRow(
                    icon: "paintbrush",
                    title: "Theme",
                    subtitle: "Light, dark, or system"
                ) {
                    AppearanceSettingsView()
                }
            }
        }
    }

    private var profileSection: some View {
        section("Profile & Security") {
            SettingsRowGroup {
                SettingsNavRow(
                    icon: "person.crop.circle",
                    title: auth.currentUser?.displayName ?? "Profile & Password",
                    subtitle: "@\(auth.currentUser?.username ?? "user")"
                ) {
                    ProfileSecuritySettingsView(vm: viewModel)
                        .environmentObject(auth)
                }
                SettingsRowDivider()
                SettingsActionRow(
                    icon: "rectangle.portrait.and.arrow.right",
                    title: "Sign Out",
                    tint: .down
                ) {
                    auth.logout()
                }
            }
        }
    }

    private var footer: some View {
        VStack(spacing: 4) {
            Text("Investa • Version 0.1.0")
                .appFont(.caption2.bold())
                .foregroundStyle(.secondary)
            Text("FastAPI + Next.js + SwiftUI")
                .appFont(.system(size: 10))
                .foregroundStyle(.secondary.opacity(0.7))
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 4)
    }

    // MARK: - Helpers

    @ViewBuilder
    private func section<C: View>(_ title: String, @ViewBuilder content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel(title: title)
                .padding(.leading, 4)
            content()
        }
    }
}
