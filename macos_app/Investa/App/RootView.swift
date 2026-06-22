import SwiftUI

/// Top-level router: shows a spinner while restoring the session, then either
/// the login screen or the dashboard.
struct RootView: View {
    @EnvironmentObject private var auth: AuthViewModel
    @StateObject private var appState = AppState()

    var body: some View {
        Group {
            switch auth.state {
            case .checking:
                ProgressView("Loading…")
                    .macMinSize(width: 420, height: 320)
            case .loggedOut:
                LoginView()
            case .loggedIn:
                MainView()
                    .environmentObject(appState)
            }
        }
        // App-wide typography bump: the UI is caption-heavy and runs small, and
        // there's room to spare, so lift every semantic font one Dynamic Type
        // step (≈ +13%). Fixed `.system(size:)` accents are scaled to match in
        // their call sites so badges keep pace with the body text.
        .dynamicTypeSize(.xLarge)
        .task {
            await auth.restoreSession()
        }
    }
}
