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

    private func send<T: Decodable>(_ request: URLRequest) async throws -> T {
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            if error is CancellationError || (error as? URLError)?.code == .cancelled {
                throw CancellationError()
            }
            print("APIClient transport error for \(request.url?.absoluteString ?? "unknown"): \(error)")
            throw APIError.transport(underlying: error)
        }

        guard let http = response as? HTTPURLResponse else {
            throw APIError.http(status: -1, detail: nil)
        }

        if http.statusCode == 401 {
            NotificationCenter.default.post(name: .authExpired, object: nil)
            throw APIError.unauthorized
        }
        guard (200..<300).contains(http.statusCode) else {
            throw APIError.http(status: http.statusCode, detail: Self.detail(from: data))
        }

        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            throw APIError.decoding(underlying: error)
        }
    }

    /// Extract FastAPI's `{"detail": "..."}` error message when present.
    private static func detail(from data: Data) -> String? {
        guard
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return obj["detail"] as? String
    }
}
