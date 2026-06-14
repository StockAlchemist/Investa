import Foundation

enum APIError: LocalizedError {
    /// HTTP 401 — token missing/expired/invalid. Triggers a logout.
    case unauthorized
    /// A non-2xx response other than 401. Carries status and any `detail` text.
    case http(status: Int, detail: String?)
    /// Networking failure (no connection, backend not running, timeout, …).
    case transport(underlying: Error)
    /// Response body could not be decoded into the expected type.
    case decoding(underlying: Error)
    /// The configured base URL is malformed.
    case invalidURL

    var errorDescription: String? {
        switch self {
        case .unauthorized:
            return "Your session has expired. Please log in again."
        case .http(let status, let detail):
            return detail ?? "Request failed (HTTP \(status))."
        case .transport:
            return "Couldn't reach the Investa backend. Is the server running on the configured address?"
        case .decoding:
            return "Received an unexpected response from the server."
        case .invalidURL:
            return "The server address is invalid."
        }
    }
}
