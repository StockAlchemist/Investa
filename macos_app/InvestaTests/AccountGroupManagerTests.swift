import XCTest
@testable import Investa

@MainActor
final class AccountGroupManagerTests: XCTestCase {
    private var testSession: URLSession!
    private var client: APIClient!
    private var viewModel: SettingsViewModel!

    override func setUp() {
        super.setUp()
        MockURLProtocol.requestCount = 0
        MockURLProtocol.requestHandler = nil

        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockURLProtocol.self]
        testSession = URLSession(configuration: config)
        client = APIClient(session: testSession)
        viewModel = SettingsViewModel(api: client)
    }

    override func tearDown() {
        MockURLProtocol.requestHandler = nil
        testSession = nil
        client = nil
        viewModel = nil
        super.tearDown()
    }

    func testAccountGroupOrderingLogic() {
        var groups: [String: [String]] = [
            "Retirement": ["Roth IRA", "401k"],
            "Taxable": ["Brokerage"],
            "Crypto": ["Coinbase"]
        ]
        var order = ["Taxable", "Retirement", "Crypto"]

        // 1. Swap order
        order.swapAt(0, 1) // -> ["Retirement", "Taxable", "Crypto"]
        XCTAssertEqual(order, ["Retirement", "Taxable", "Crypto"])

        // 2. Rename group in place
        let oldName = "Taxable"
        let newName = "Taxable Main"
        let accounts = groups.removeValue(forKey: oldName) ?? []
        groups[newName] = accounts
        if let idx = order.firstIndex(of: oldName) {
            order[idx] = newName
        }

        XCTAssertEqual(order, ["Retirement", "Taxable Main", "Crypto"])
        XCTAssertNil(groups["Taxable"])
        XCTAssertEqual(groups["Taxable Main"], ["Brokerage"])

        // 3. Delete group
        groups.removeValue(forKey: "Crypto")
        order = order.filter { $0 != "Crypto" }

        XCTAssertEqual(order, ["Retirement", "Taxable Main"])
        XCTAssertEqual(groups.keys.count, 2)
    }

    private static func extractBodyData(from request: URLRequest) -> Data? {
        if let body = request.httpBody {
            return body
        }
        guard let stream = request.httpBodyStream else { return nil }
        stream.open()
        defer { stream.close() }
        var data = Data()
        let bufferSize = 1024
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
        defer { buffer.deallocate() }
        while stream.hasBytesAvailable {
            let read = stream.read(buffer, maxLength: bufferSize)
            if read > 0 {
                data.append(buffer, count: read)
            } else {
                break
            }
        }
        return data.isEmpty ? nil : data
    }

    func testViewModelUpdateGroupsRequest() async throws {
        var capturedBody: [String: Any]?
        MockURLProtocol.requestHandler = { request in
            if let bodyData = Self.extractBodyData(from: request) {
                capturedBody = try? JSONSerialization.jsonObject(with: bodyData) as? [String: Any]
            }

            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            let data = """
            { "status": "ok", "message": "Saved" }
            """.data(using: .utf8)!
            return (response, data)
        }

        let newGroups = [
            "Growth": ["Account A", "Account B"],
            "Dividends": ["Account C"]
        ]
        let newOrder = ["Growth", "Dividends"]

        await viewModel.updateGroups(newGroups, order: newOrder)

        XCTAssertEqual(viewModel.status, "Groups saved.")
        XCTAssertNotNil(capturedBody)
        let groupsPayload = capturedBody?["account_groups"] as? [String: [String]]
        let orderPayload = capturedBody?["account_group_order"] as? [String]

        XCTAssertEqual(groupsPayload?["Growth"], ["Account A", "Account B"])
        XCTAssertEqual(groupsPayload?["Dividends"], ["Account C"])
        XCTAssertEqual(orderPayload, ["Growth", "Dividends"])
    }

    func testSettingsAllAccountsComputation() {
        let json = """
        {
            "account_groups": {
                "B Group": ["Account 2", "Account 3"],
                "A Group": ["Account 1", "Account 2"]
            },
            "account_group_order": ["A Group", "B Group"]
        }
        """.data(using: .utf8)!

        let settings = try? JSONDecoder().decode(AppSettings.self, from: json)
        XCTAssertNotNil(settings)

        // All accounts should be deduplicated in group order: "Account 1", "Account 2", "Account 3"
        let all = settings?.allAccounts ?? []
        XCTAssertEqual(all, ["Account 1", "Account 2", "Account 3"])
    }
}
