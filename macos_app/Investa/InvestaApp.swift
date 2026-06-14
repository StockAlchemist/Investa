import SwiftUI

@main
struct InvestaApp: App {
    @StateObject private var auth = AuthViewModel()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(auth)
                .frame(minWidth: 900, minHeight: 600)
        }
        .windowResizability(.contentMinSize)
        .commands {
            // Native menu bar entries; broadcast intents the dashboard observes.
            CommandGroup(after: .toolbar) {
                Button("Refresh") {
                    NotificationCenter.default.post(name: .refreshRequested, object: nil)
                }
                .keyboardShortcut("r", modifiers: .command)
                .disabled(auth.currentUser == nil)
            }
            CommandGroup(replacing: .appInfo) {
                Button("About Investa") {
                    NSApplication.shared.orderFrontStandardAboutPanel(nil)
                }
            }
            CommandMenu("Account") {
                if let user = auth.currentUser {
                    Text("Signed in as \(user.displayName)")
                    Divider()
                }
                Button("Log Out") { auth.logout() }
                    .keyboardShortcut("l", modifiers: [.command, .shift])
                    .disabled(auth.currentUser == nil)
            }
        }
    }
}

extension Notification.Name {
    /// Posted by the Refresh menu command (⌘R); the dashboard reloads on it.
    static let refreshRequested = Notification.Name("investa.refreshRequested")
}
