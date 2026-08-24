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
        // App-wide typography bump on iOS: the UI is caption-heavy and runs
        // small, so lift every semantic font one Dynamic Type step (≈ +13%).
        // macOS ignores this modifier entirely — it has no Dynamic Type — and
        // gets its bump from `\.appFontScale` instead (see AppFont.swift).
        .dynamicTypeSize(.xLarge)
        .toastOverlay()
        .task {
            await auth.restoreSession()
        }
    }
}
