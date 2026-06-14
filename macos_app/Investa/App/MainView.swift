import SwiftUI

enum AppSection: String, CaseIterable, Identifiable {
    case dashboard = "Dashboard"
    case transactions = "Transactions"
    case dividends = "Dividends"
    case capitalGains = "Capital Gains"
    case watchlist = "Watchlist"

    var id: String { rawValue }
    var icon: String {
        switch self {
        case .dashboard: return "chart.pie"
        case .transactions: return "list.bullet.rectangle"
        case .dividends: return "dollarsign.circle"
        case .capitalGains: return "arrow.up.right"
        case .watchlist: return "star"
        }
    }
}

/// Sidebar-based shell hosting the feature tabs. Replaces the standalone
/// dashboard window once the user is logged in.
struct MainView: View {
    @EnvironmentObject private var auth: AuthViewModel
    @EnvironmentObject private var appState: AppState
    @State private var selection: AppSection = .dashboard

    var body: some View {
        NavigationSplitView {
            List(AppSection.allCases, selection: $selection) { section in
                Label(section.rawValue, systemImage: section.icon)
                    .tag(section)
            }
            .navigationSplitViewColumnWidth(min: 180, ideal: 200, max: 240)
            .safeAreaInset(edge: .bottom) { accountFooter }
        } detail: {
            switch selection {
            case .dashboard: DashboardView()
            case .transactions: TransactionsView()
            case .dividends: DividendsView()
            case .capitalGains: CapitalGainsView()
            case .watchlist: WatchlistView()
            }
        }
    }

    private var accountFooter: some View {
        VStack(spacing: 0) {
            Divider()
            Menu {
                if let user = auth.currentUser {
                    Text("Signed in as \(user.displayName)")
                    Divider()
                }
                Button("Refresh") {
                    NotificationCenter.default.post(name: .refreshRequested, object: nil)
                }
                Button("Log Out") { auth.logout() }
            } label: {
                Label(auth.currentUser?.displayName ?? "Account", systemImage: "person.crop.circle")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .menuStyle(.borderlessButton)
            .padding(10)
        }
    }
}
