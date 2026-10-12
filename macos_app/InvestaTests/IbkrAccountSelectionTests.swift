import XCTest
@testable import Investa

@MainActor
final class IbkrAccountSelectionTests: XCTestCase {
    private func model() -> TransactionsViewModel {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockURLProtocol.self]
        return TransactionsViewModel(api: APIClient(session: URLSession(configuration: configuration)))
    }

    func testApprovalSendsOnlyApprovedRowsAccountChoices() async throws {
        var submitted: [String: Any]?
        MockURLProtocol.requestHandler = { request in
            if request.url!.path.hasSuffix("/approve") {
                var body = request.httpBody ?? Data()
                if body.isEmpty, let stream = request.httpBodyStream {
                    stream.open()
                    defer { stream.close() }
                    var buffer = [UInt8](repeating: 0, count: 4096)
                    while stream.hasBytesAvailable {
                        let count = stream.read(&buffer, maxLength: buffer.count)
                        if count <= 0 { break }
                        body.append(contentsOf: buffer.prefix(count))
                    }
                }
                submitted = try JSONSerialization.jsonObject(with: body) as? [String: Any]
                return (HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!, Data("{\"status\":\"success\"}".utf8))
            }
            return (HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!, Data("[]".utf8))
        }
        defer { MockURLProtocol.requestHandler = nil }
        let viewModel = model()
        viewModel.pendingAccountSelections = [701: "Retirement", 702: "Taxable"]
        await viewModel.approvePending([701])
        await Task.yield()
        for _ in 0..<100 where viewModel.isLoading { try? await Task.sleep(nanoseconds: 10_000_000) }
        XCTAssertEqual(submitted?["ids"] as? [Int], [701])
        XCTAssertEqual(submitted?["accounts"] as? [String: String], ["701": "Retirement"])
        XCTAssertFalse(viewModel.isProcessingPending)
    }

    func testBulkApprovalPayloadKeepsIndependentAccounts() throws {
        let payload = IbkrApprovalInput(ids: [701, 702], accounts: ["701": "Retirement", "702": "Taxable"])
        let json = try JSONSerialization.jsonObject(with: JSONEncoder().encode(payload)) as? [String: Any]
        XCTAssertEqual(json?["ids"] as? [Int], [701, 702])
        XCTAssertEqual(json?["accounts"] as? [String: String], ["701": "Retirement", "702": "Taxable"])
    }

    func testFailedApprovalRetainsChoicesAndRows() async {
        MockURLProtocol.requestHandler = { _ in throw URLError(.networkConnectionLost) }
        defer { MockURLProtocol.requestHandler = nil }
        let viewModel = model()
        viewModel.pendingAccountSelections = [701: "Retirement"]
        await viewModel.approvePending([701])
        XCTAssertEqual(viewModel.pendingAccountSelections[701], "Retirement")
        XCTAssertNotNil(viewModel.errorMessage)
        XCTAssertFalse(viewModel.isProcessingPending)
    }
}
