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
                    .frame(minWidth: 420, minHeight: 320)
            case .loggedOut:
                LoginView()
            case .loggedIn:
                MainView()
                    .environmentObject(appState)
            }
        }
        .task {
            await auth.restoreSession()
        }
    }
}
