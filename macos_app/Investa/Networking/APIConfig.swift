import Foundation

/// Resolves and persists the backend base URL. Defaults to the local dev server.
/// Mirrors the web client's `getApiBaseUrl()` default of `http://localhost:8000/api`.
enum APIConfig {
    private static let defaultsKey = "investa.api.baseURL"
    static let fallbackBaseURL = "http://localhost:8000/api"

    static var baseURL: String {
        get {
            let stored = UserDefaults.standard.string(forKey: defaultsKey)
            var value = (stored?.isEmpty == false ? stored! : fallbackBaseURL)
            if value.contains("localhost") {
                value = value.replacingOccurrences(of: "localhost", with: "127.0.0.1")
            }
            // Strip a trailing slash so path joining is predictable.
            return value.hasSuffix("/") ? String(value.dropLast()) : value
        }
        set {
            UserDefaults.standard.set(newValue, forKey: defaultsKey)
        }
    }
}
