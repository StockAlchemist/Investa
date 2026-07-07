import SwiftUI

@main
struct InvestaApp: App {
    @StateObject private var auth = AuthViewModel()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(auth)
                .tint(Theme.brand)
                // Force Gregorian app-wide so chart date axes (and SwiftUI date
                // rendering) don't show the Buddhist era under a Thai locale.
                // Financial dates are stored as ISO yyyy-MM-dd.
                .gregorianCalendar()
                // iPhone-only: shrink semantic text one notch (iPad/macOS unchanged).
                .iPhoneTextScale()
                .macMinSize(width: 900, height: 600)
        }
        #if os(macOS)
        .windowResizability(.contentMinSize)
        .commands {
            // Native menu bar entries; broadcast intents the dashboard observes.
            CommandGroup(after: .toolbar) {
                Button("Refresh") {
                    NotificationCenter.default.post(name: .refreshRequested, object: nil)
                }
                .keyboardShortcut("r", modifiers: .command)
                .disabled(auth.currentUser == nil)

                Button("Command Palette…") {
                    NotificationCenter.default.post(name: .commandPalette, object: nil)
                }
                .keyboardShortcut("k", modifiers: .command)
                .disabled(auth.currentUser == nil)

                Divider()

                Button("Settings…") {
                    NotificationCenter.default.post(name: .openSettings, object: nil)
                }
                .keyboardShortcut(",", modifiers: .command)

                Button("Toggle Dark Mode") {
                    NotificationCenter.default.post(name: .toggleDarkMode, object: nil)
                }
                .keyboardShortcut("d", modifiers: [.command, .shift])
            }
            CommandGroup(replacing: .appInfo) {
                Button("About Investa") {
                    NSApplication.shared.orderFrontStandardAboutPanel(nil)
                }
            }

            // "Go" menu — navigate to any sidebar section via keyboard shortcuts.
            CommandMenu("Go") {
                Section("Portfolio") {
                    goButton(.performance,   shortcut: "1")
                    goButton(.allocation,    shortcut: "2")
                    goButton(.assetChange,   shortcut: "3")
                    goButton(.transactions,  shortcut: "4")
                    goButton(.dividend,      shortcut: "5")
                    goButton(.capitalGains,  shortcut: "6")
                }
                Section("Discover") {
                    goButton(.market,    shortcut: "7")
                    goButton(.watchlist, shortcut: "8")
                    goButton(.markets,   shortcut: "9")
                    goButton(.aiReview,  shortcut: "0")
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
        #endif
    }

    /// Creates a menu button that navigates to a sidebar section via notification.
    private func goButton(_ section: AppSection, shortcut: Character) -> some View {
        Button(section.rawValue) {
            NotificationCenter.default.post(name: .navigateToSection, object: section)
        }
        .keyboardShortcut(KeyEquivalent(shortcut), modifiers: .command)
        .disabled(auth.currentUser == nil)
    }
}

extension Notification.Name {
    /// Posted by the Refresh menu command (⌘R); the dashboard reloads on it.
    static let refreshRequested = Notification.Name("investa.refreshRequested")
    /// Posted by ⌘K to open the command palette.
    static let commandPalette = Notification.Name("investa.commandPalette")
    /// Posted by the Go menu (⌘1–⌘0) to navigate to a sidebar section.
    static let navigateToSection = Notification.Name("investa.navigateToSection")
    /// Posted by ⌘, to open the Settings sheet.
    static let openSettings = Notification.Name("investa.openSettings")
    /// Posted by ⇧⌘D to toggle dark/light mode.
    static let toggleDarkMode = Notification.Name("investa.toggleDarkMode")
}
