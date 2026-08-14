import XCTest
@testable import Investa

/// Custom URLProtocol to intercept and mock network requests for APIClient testing.
final class MockURLProtocol: URLProtocol {
    typealias RequestHandler = (URLRequest) throws -> (HTTPURLResponse, Data)

    static var requestHandler: RequestHandler?
    static var requestCount = 0

    override class func canInit(with request: URLRequest) -> Bool {
        return true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        return request
    }

    override func startLoading() {
        Self.requestCount += 1
        guard let handler = Self.requestHandler else {
            client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
            return
        }

        do {
            let (response, data) = try handler(request)
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}

final class APIClientRetryTests: XCTestCase {
    private var testSession: URLSession!
    private var client: APIClient!

    override func setUp() {
        super.setUp()
        MockURLProtocol.requestCount = 0
        MockURLProtocol.requestHandler = nil

        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockURLProtocol.self]
        testSession = URLSession(configuration: config)
        client = APIClient(session: testSession)
    }

    override func tearDown() {
        MockURLProtocol.requestHandler = nil
        testSession = nil
        client = nil
        super.tearDown()
    }

    func testSuccessfulGetRequestWithoutRetry() async throws {
        MockURLProtocol.requestHandler = { request in
            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            let data = """
            { "status": "ok", "message": "Success" }
            """.data(using: .utf8)!
            return (response, data)
        }

        let result: StatusResponse = try await client.get("/test")
        XCTAssertEqual(result.status, "ok")
        XCTAssertEqual(MockURLProtocol.requestCount, 1)
    }

    func testRetryOnTransient503SucceedsOnSecondAttempt() async throws {
        var attempts = 0
        MockURLProtocol.requestHandler = { request in
            attempts += 1
            if attempts == 1 {
                let response = HTTPURLResponse(
                    url: request.url!,
                    statusCode: 503,
                    httpVersion: nil,
                    headerFields: ["Content-Type": "application/json"]
                )!
                let data = """
                { "detail": "Service Temporarily Unavailable" }
                """.data(using: .utf8)!
                return (response, data)
            } else {
                let response = HTTPURLResponse(
                    url: request.url!,
                    statusCode: 200,
                    httpVersion: nil,
                    headerFields: ["Content-Type": "application/json"]
                )!
                let data = """
                { "status": "ok" }
                """.data(using: .utf8)!
                return (response, data)
            }
        }

        let result: StatusResponse = try await client.get("/test_retry")
        XCTAssertEqual(result.status, "ok")
        XCTAssertEqual(attempts, 2)
    }

    func testFailFastOnNonRetryable400Error() async throws {
        MockURLProtocol.requestHandler = { request in
            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: 400,
                httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            let data = """
            { "detail": "Invalid input format" }
            """.data(using: .utf8)!
            return (response, data)
        }

        do {
            let _: StatusResponse = try await client.get("/bad_request")
            XCTFail("Expected 400 error to throw")
        } catch let APIError.http(status, detail) {
            XCTAssertEqual(status, 400)
            XCTAssertEqual(detail, "Invalid input format")
            XCTAssertEqual(MockURLProtocol.requestCount, 1, "Should not retry 400 client error")
        }
    }

    func test401UnauthorizedPostsAuthExpiredNotification() async throws {
        let expectation = expectation(description: "authExpired notification received")
        let cancellable = NotificationCenter.default.publisher(for: .authExpired)
            .sink { _ in
                expectation.fulfill()
            }

        MockURLProtocol.requestHandler = { request in
            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: 401,
                httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            let data = """
            { "detail": "Token expired" }
            """.data(using: .utf8)!
            return (response, data)
        }

        do {
            let _: StatusResponse = try await client.get("/protected")
            XCTFail("Expected unauthorized error to throw")
        } catch APIError.unauthorized {
            // expected
        }

        await fulfillment(of: [expectation], timeout: 2.0)
        XCTAssertEqual(MockURLProtocol.requestCount, 1)
        _ = cancellable
    }

    func testRetryOnTransientTimeoutError() async throws {
        var attempts = 0
        MockURLProtocol.requestHandler = { request in
            attempts += 1
            if attempts == 1 {
                throw URLError(.timedOut)
            } else {
                let response = HTTPURLResponse(
                    url: request.url!,
                    statusCode: 200,
                    httpVersion: nil,
                    headerFields: ["Content-Type": "application/json"]
                )!
                let data = """
                { "status": "recovered" }
                """.data(using: .utf8)!
                return (response, data)
            }
        }

        let result: StatusResponse = try await client.get("/timeout_test")
        XCTAssertEqual(result.status, "recovered")
        XCTAssertEqual(attempts, 2)
    }

    func testExhaustRetriesOnPersistent503Error() async throws {
        MockURLProtocol.requestHandler = { request in
            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: 503,
                httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            let data = """
            { "detail": "Backend unavailable" }
            """.data(using: .utf8)!
            return (response, data)
        }

        do {
            let _: StatusResponse = try await client.get("/persistent_failure")
            XCTFail("Expected persistent 503 error to throw")
        } catch let APIError.http(status, detail) {
            XCTAssertEqual(status, 503)
            XCTAssertEqual(detail, "Backend unavailable")
            XCTAssertEqual(MockURLProtocol.requestCount, 3, "Expected 3 attempts (1 initial + 2 retries)")
        }
    }
}
