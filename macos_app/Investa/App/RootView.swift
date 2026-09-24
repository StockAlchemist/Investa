import SwiftUI

/// Top-level router: shows a spinner while restoring the session, then either
/// the login screen or the dashboard.
struct RootView: View {
    @EnvironmentObject private var auth: AuthViewModel

    var body: some View {
        Group {
            switch auth.state {
            case .checking:
                ProgressView("Loading…")
                    .macMinSize(width: 420, height: 320)
            case .loggedOut:
                LoginView()
            case .loggedIn(let user):
                SignedInRoot()
                    .id(user.id)
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

/// The signed-in tree, owning one `AppState` per session.
///
/// `AppState` holds the user's accounts, selection, currency and headline
/// figures. It used to live on `RootView`, above the login switch, so it
/// outlived a logout: the next user to sign in inherited all of it — and with
/// it `didLoadSettings`, so their own settings were never fetched and their
/// dashboard was filtered by the previous user's account names.
private struct SignedInRoot: View {
    @StateObject private var appState = AppState()

    var body: some View {
        MainView()
            .environmentObject(appState)
    }
}
