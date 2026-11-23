import Foundation

/// Resolves and persists the backend base URL. Defaults to the local dev server.
/// Mirrors the web client's `getApiBaseUrl()` default of `http://localhost:8800/api`.
enum APIConfig {
    private static let defaultsKey = "investa.api.baseURL"
    private static let savedServersKey = "investa.api.savedServers"
    private static let portMigrationKey = "investa.api.migratedTo8800"
    static let fallbackBaseURL = "http://localhost:8800/api"

    /// The backend moved from port 8000 to 8800 (Nov 2026: 8000 is Tuva's
    /// daemon's port on the server Mac). An address saved before then would
    /// stop connecting, so rewrite it once: plain-HTTP `…:8000/api` becomes
    /// `…:8800/api`. HTTPS addresses carry no port (Tailscale Serve maps
    /// `/api` itself) and are left alone. Runs once per device, so a backend
    /// someone deliberately points at 8000 afterwards stays where they put it.
    static func migrateLegacyPortIfNeeded(_ defaults: UserDefaults = .standard) {
        guard !defaults.bool(forKey: portMigrationKey) else { return }
        defaults.set(true, forKey: portMigrationKey)
        if let stored = defaults.string(forKey: defaultsKey) {
            defaults.set(migratedFromLegacyPort(stored), forKey: defaultsKey)
        }
        if let data = defaults.data(forKey: savedServersKey),
           var servers = try? JSONDecoder().decode([SavedServer].self, from: data) {
            for i in servers.indices {
                let url = migratedFromLegacyPort(servers[i].url)
                // An unnamed entry is labelled host:port; keep the label true.
                if servers[i].name == ServerStore.defaultName(for: servers[i].url) {
                    servers[i].name = ServerStore.defaultName(for: url)
                }
                servers[i].url = url
            }
            if let encoded = try? JSONEncoder().encode(servers) {
                defaults.set(encoded, forKey: savedServersKey)
            }
        }
    }

    /// `http://host:8000/api` → `http://host:8800/api`; anything else unchanged.
    static func migratedFromLegacyPort(_ url: String) -> String {
        let value = normalized(url)
        guard var comps = URLComponents(string: value),
              comps.scheme?.lowercased() == "http", comps.port == 8000,
              comps.path == "/api" || comps.path.hasPrefix("/api/")
        else { return url }
        comps.port = 8800
        return comps.string ?? url
    }

    static var baseURL: String {
        get {
            let stored = UserDefaults.standard.string(forKey: defaultsKey).map(normalized)
            return stored?.isEmpty == false ? stored! : normalized(fallbackBaseURL)
        }
        set {
            UserDefaults.standard.set(newValue, forKey: defaultsKey)
        }
    }

    /// The form every URL is stored and compared in: trimmed, `localhost` as
    /// `127.0.0.1`, no trailing slash (so path joining is predictable).
    static func normalized(_ url: String) -> String {
        var value = url.trimmingCharacters(in: .whitespacesAndNewlines)
        if value.contains("localhost") {
            value = value.replacingOccurrences(of: "localhost", with: "127.0.0.1")
        }
        while value.hasSuffix("/") { value.removeLast() }
        return value
    }
}
