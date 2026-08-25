import Foundation

extension Notification.Name {
    /// Posted when the backend rejects the token (HTTP 401). The auth layer
    /// listens for this to clear credentials and return to the login screen.
    /// Mirrors the web client's `auth:expired` event.
    static let authExpired = Notification.Name("investa.authExpired")
}

/// Async HTTP client for the Investa FastAPI backend.
///
/// Stateless aside from reading the bearer token from `KeychainStore` on each
/// request, so it is safe to share. All methods are `async` and throw `APIError`.
final class APIClient: Sendable {
    static let shared = APIClient()

    private let session: URLSession
    private let decoder: JSONDecoder

    init(session: URLSession = .shared) {
        self.session = session
        self.decoder = JSONDecoder()
    }

    // MARK: - Public requests

    /// GET a JSON endpoint and decode it into `T`.
    func get<T: Decodable>(_ path: String, query: [URLQueryItem] = []) async throws -> T {
        let request = try makeRequest(path: path, method: "GET", query: query)
        return try await send(request)
    }

    /// POST `application/x-www-form-urlencoded` (used by the OAuth2 login route).
    func postForm<T: Decodable>(_ path: String, fields: [String: String]) async throws -> T {
        var request = try makeRequest(path: path, method: "POST", query: [])
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        var comps = URLComponents()
        comps.queryItems = fields.map { URLQueryItem(name: $0.key, value: $0.value) }
        request.httpBody = comps.percentEncodedQuery?.data(using: .utf8)
        return try await send(request)
    }

    /// POST a `multipart/form-data` upload of a single file (used by document parsing).
    func postMultipart<T: Decodable>(_ path: String, fileURL: URL, fieldName: String = "file") async throws -> T {
        var request = try makeRequest(path: path, method: "POST", query: [])
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 120
        let fileData: Data
        do { fileData = try Data(contentsOf: fileURL) } catch { throw APIError.transport(underlying: error) }
        var body = Data()
        func append(_ s: String) { body.append(s.data(using: .utf8)!) }
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"\(fieldName)\"; filename=\"\(fileURL.lastPathComponent)\"\r\n")
        append("Content-Type: application/octet-stream\r\n\r\n")
        body.append(fileData)
        append("\r\n--\(boundary)--\r\n")
        request.httpBody = body
        return try await send(request)
    }

    /// Generic JSON request for POST/PUT/DELETE with an optional `Encodable` body.
    func send<T: Decodable>(
        method: String, path: String, query: [URLQueryItem] = [], body: (any Encodable)? = nil
    ) async throws -> T {
        var request = try makeRequest(path: path, method: method, query: query)
        if let body {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            do {
                request.httpBody = try JSONEncoder().encode(body)
            } catch {
                throw APIError.decoding(underlying: error)
            }
        }
        return try await send(request)
    }

    // MARK: - Helpers

    /// Build a repeated-value query item list (FastAPI parses repeats into a list).
    static func arrayQuery(_ name: String, _ values: [String]?) -> [URLQueryItem] {
        guard let values, !values.isEmpty else { return [] }
        return values.map { URLQueryItem(name: name, value: $0) }
    }

    private func makeRequest(path: String, method: String, query: [URLQueryItem]) throws -> URLRequest {
        let trimmed = path.hasPrefix("/") ? String(path.dropFirst()) : path
        guard var comps = URLComponents(string: APIConfig.baseURL + "/" + trimmed) else {
            throw APIError.invalidURL
        }
        if !query.isEmpty {
            comps.queryItems = (comps.queryItems ?? []) + query
        }
        guard let url = comps.url else { throw APIError.invalidURL }

        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if let token = KeychainStore.loadToken() {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        
        let isSlowEndpoint = trimmed.hasPrefix("ai") || trimmed.hasPrefix("sync") || trimmed.hasPrefix("stock-analysis") || trimmed.hasPrefix("screener") || trimmed.hasPrefix("chat") || trimmed.hasPrefix("portfolio/ai_review")
        request.timeoutInterval = isSlowEndpoint ? 300 : 60
        
        return request
    }

    // MARK: - Core Execution with Exponential Backoff Retries

    private func send<T: Decodable>(_ request: URLRequest, maxAttempts: Int = 3, baseDelay: TimeInterval = 0.35) async throws -> T {
        var lastError: Error?
        var lastResponse: HTTPURLResponse?

        for attempt in 0..<maxAttempts {
            if attempt > 0 {
                let jitter = Double.random(in: 0.02...0.08)
                let delay = baseDelay * pow(2.0, Double(attempt - 1)) + jitter
                try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                if Task.isCancelled {
                    throw CancellationError()
                }
            }

            do {
                let (data, response) = try await session.data(for: request)
                guard let http = response as? HTTPURLResponse else {
                    let err = APIError.http(status: -1, detail: nil)
                    lastError = err
                    break
                }
                lastResponse = http

                if http.statusCode == 401 {
                    NotificationCenter.default.post(name: .authExpired, object: nil)
                    throw APIError.unauthorized
                }

                // If transient server error (502, 503, 504, 429) and attempts remain, retry.
                if isRetryableStatusCode(http.statusCode) && attempt < maxAttempts - 1 {
                    continue
                }

                guard (200..<300).contains(http.statusCode) else {
                    let detail = Self.detail(from: data)
                    let err = APIError.http(status: http.statusCode, detail: detail)
                    // If final attempt was a 50x server error, present a toast.
                    if (500...599).contains(http.statusCode) {
                        presentToast(for: err, httpStatus: http.statusCode)
                    }
                    throw err
                }

                do {
                    return try decoder.decode(T.self, from: data)
                } catch {
                    throw APIError.decoding(underlying: error)
                }
            } catch {
                if error is CancellationError || (error as? URLError)?.code == .cancelled {
                    throw CancellationError()
                }
                if let apiErr = error as? APIError {
                    switch apiErr {
                    case .unauthorized, .decoding, .invalidURL:
                        throw apiErr
                    case .http(let status, _):
                        if !isRetryableStatusCode(status) || attempt == maxAttempts - 1 {
                            throw apiErr
                        }
                    case .transport:
                        break
                    }
                }

                lastError = error
                // If it's not a retryable transport error or this was the last attempt, break and report.
                if !isRetryableTransportError(error) || attempt == maxAttempts - 1 {
                    break
                }
            }
        }

        let finalError: Error = lastError ?? APIError.http(status: lastResponse?.statusCode ?? -1, detail: nil)
        let mappedError: APIError
        if let apiErr = finalError as? APIError {
            mappedError = apiErr
        } else {
            mappedError = APIError.transport(underlying: finalError)
        }

        presentToast(for: mappedError, httpStatus: lastResponse?.statusCode)
        throw mappedError
    }

    private func isRetryableStatusCode(_ code: Int) -> Bool {
        code == 502 || code == 503 || code == 504 || code == 429
    }

    private func isRetryableTransportError(_ error: Error) -> Bool {
        if let urlError = error as? URLError {
            switch urlError.code {
            case .timedOut,
                 .cannotConnectToHost,
                 .networkConnectionLost,
                 .notConnectedToInternet,
                 .dnsLookupFailed,
                 .cannotFindHost,
                 .resourceUnavailable,
                 .internationalRoamingOff,
                 .dataNotAllowed:
                return true
            default:
                return false
            }
        }
        return true
    }

    private func presentToast(for error: APIError, httpStatus: Int?) {
        let message: String
        let style: ToastStyle

        switch error {
        case .unauthorized:
            return // handled by authExpired notification
        case .invalidURL:
            message = "Invalid server URL configured."
            style = .error
        case .decoding:
            message = "Received malformed data from server."
            style = .error
        case .http(let status, let detail):
            if status == 502 || status == 503 || status == 504 {
                message = detail ?? "Investa server is temporarily unavailable (HTTP \(status))."
                style = .warning
            } else {
                message = detail ?? "Request failed (HTTP \(status))."
                style = .error
            }
        case .transport(let underlying):
            if let urlErr = underlying as? URLError {
                if urlErr.code == .notConnectedToInternet {
                    message = "You appear to be offline. Please check your internet connection."
                    style = .warning
                } else if urlErr.code == .timedOut {
                    message = "Server request timed out. Retrying may succeed."
                    style = .warning
                } else {
                    message = "Cannot reach server: \(urlErr.localizedDescription)"
                    style = .error
                }
            } else {
                message = "Network connection error. Is the Investa backend running?"
                style = .error
            }
        }

        DispatchQueue.main.async {
            ToastManager.shared.show(message: message, style: style)
        }
    }

    /// Extract FastAPI's `{"detail": "..."}` error message when present, falling
    /// back to `message` for the routes that answer in that shape (IBKR sync).
    private static func detail(from data: Data) -> String? {
        guard
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return obj["detail"] as? String ?? obj["message"] as? String
    }
}
