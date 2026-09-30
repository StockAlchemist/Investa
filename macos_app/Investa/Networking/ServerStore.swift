import Foundation

/// A backend the user has named and kept, so moving between them — home LAN,
/// Tailscale, a Mac on the desk — is one tap instead of retyping an address.
struct SavedServer: Codable, Identifiable, Hashable {
    var id = UUID()
    var name: String
    var url: String
}

/// The saved-server list and the active backend, observable so the settings
/// card, the login sheet and the control bar's quick switcher agree.
///
/// Switching changes only `APIConfig.baseURL`. The session token stays, so two
/// addresses of the same backend (LAN and Tailscale) switch without signing in
/// again; a different backend rejects the token and the usual 401 path shows
/// the login screen.
@MainActor
final class ServerStore: ObservableObject {
    static let shared = ServerStore()

    private static let defaultsKey = "investa.api.savedServers"

    @Published private(set) var servers: [SavedServer]
    @Published private(set) var activeURL: String

    private init() {
        let data = UserDefaults.standard.data(forKey: Self.defaultsKey)
        servers = data.flatMap { try? JSONDecoder().decode([SavedServer].self, from: $0) } ?? []
        activeURL = APIConfig.baseURL
    }

    /// The saved entry for the backend in use, if it has one.
    var activeServer: SavedServer? { servers.first { isActive($0) } }

    func isActive(_ server: SavedServer) -> Bool {
        APIConfig.normalized(server.url) == activeURL
    }

    /// Points every request at `url` and reloads the app's data from it.
    func switchTo(_ url: String) {
        let target = APIConfig.normalized(url)
        guard !target.isEmpty else { return }
        APIConfig.baseURL = target
        activeURL = APIConfig.baseURL
        NotificationCenter.default.post(name: .refreshRequested, object: nil)
    }

    /// Saves `url` under `name`, or under its host when no name is given. An
    /// address already in the list is renamed rather than listed twice.
    @discardableResult
    func add(name: String, url: String) -> SavedServer? {
        let target = APIConfig.normalized(url)
        guard !target.isEmpty else { return nil }
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let label = trimmed.isEmpty ? Self.defaultName(for: target) : trimmed
        if let i = servers.firstIndex(where: { APIConfig.normalized($0.url) == target }) {
            servers[i].name = label
            persist()
            return servers[i]
        }
        let server = SavedServer(name: label, url: target)
        servers.append(server)
        persist()
        return server
    }

    func remove(_ server: SavedServer) {
        servers.removeAll { $0.id == server.id }
        persist()
    }

    /// `http://100.127.10.38:8000/api` → `100.127.10.38:8000`.
    nonisolated static func defaultName(for url: String) -> String {
        guard let comps = URLComponents(string: url), let host = comps.host else { return url }
        return comps.port.map { "\(host):\($0)" } ?? host
    }

    private func persist() {
        if let data = try? JSONEncoder().encode(servers) {
            UserDefaults.standard.set(data, forKey: Self.defaultsKey)
        }
    }
}
