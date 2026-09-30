import Foundation

/// Resolves and persists the backend base URL. Defaults to the local dev server.
/// Mirrors the web client's `getApiBaseUrl()` default of `http://localhost:8000/api`.
enum APIConfig {
    private static let defaultsKey = "investa.api.baseURL"
    static let fallbackBaseURL = "http://localhost:8000/api"

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
