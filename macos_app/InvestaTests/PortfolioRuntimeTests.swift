import XCTest
@testable import Investa

@MainActor
final class PortfolioRuntimeTests: XCTestCase {
    private func client() -> APIClient {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockURLProtocol.self]
        return APIClient(session: URLSession(configuration: config))
    }

    func testProtectedWriteRetriesWithSameReceiptKey() async throws {
        var keys: [String] = []
        MockURLProtocol.requestHandler = { request in
            keys.append(request.value(forHTTPHeaderField: "Idempotency-Key") ?? "")
            if keys.count == 1 { throw URLError(.networkConnectionLost) }
            return (HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!, Data("{\"status\":\"success\",\"id\":1}".utf8))
        }
        defer { MockURLProtocol.requestHandler = nil }
        struct Body: Encodable { let Symbol = "AAPL" }
        let _: StatusResponse = try await client().send(method: "POST", path: "/transactions", body: Body())
        XCTAssertEqual(keys.count, 2)
        XCTAssertFalse(keys[0].isEmpty)
        XCTAssertEqual(keys[0], keys[1])
    }

    func testUnprotectedWriteIsNotRetried() async {
        var count = 0
        MockURLProtocol.requestHandler = { _ in count += 1; throw URLError(.timedOut) }
        defer { MockURLProtocol.requestHandler = nil }
        do {
            let _: StatusResponse = try await client().send(method: "POST", path: "/sync/ibkr")
            XCTFail("Expected a timeout")
        } catch {}
        XCTAssertEqual(count, 1)
    }

    func testPeriodChangeDoesNotReloadPortfolioOrHiddenWidgets() async throws {
        var paths: [String] = []
        MockURLProtocol.requestHandler = { request in
            let path = request.url!.path
            paths.append(path)
            let body = path.hasSuffix("/summary") || path.hasSuffix("/summary/headline")
                ? "{\"metrics\":{\"market_value\":100}}" : "[]"
            return (HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!, Data(body.utf8))
        }
        defer { MockURLProtocol.requestHandler = nil }
        let model = DashboardViewModel(api: client())
        let visible: Set<String> = ["performanceGraph"]
        model.reload(currency: "USD", accounts: nil, period: .oneYear, showClosed: false, benchmarks: [], visibleItems: visible)
        await settled(model)
        let firstHoldings = paths.filter { $0.hasSuffix("/holdings") }.count
        model.reload(currency: "USD", accounts: nil, period: .oneDay, showClosed: false, benchmarks: [], visibleItems: visible)
        await settled(model)
        XCTAssertEqual(firstHoldings, 1)
        XCTAssertEqual(paths.filter { $0.hasSuffix("/holdings") }.count, 1)
        XCTAssertFalse(paths.contains { $0.contains("projection") || $0.contains("risk_metrics") })
    }

    func testSharedContractFixtureDecodes() throws {
        let url = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "portfolio_contract", withExtension: "json"))
        struct Fixture: Decodable { let summary: SummaryResponse; let holdings: [Holding]; let history: [PerformancePoint] }
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        XCTAssertEqual(fixture.summary.metrics?.marketValue, 125450)
        XCTAssertNil(fixture.summary.metrics?.annualizedTWR)
        XCTAssertEqual(fixture.summary.freshness?.refreshing, true)
        XCTAssertEqual(fixture.holdings.first?.symbol, "AAPL")
        XCTAssertEqual(fixture.history.first?.benchmark("S&P 500"), 10)
    }

    func testNavigationRoundTrip() throws {
        let url = ViewNavigation.url(section: .allocation, currency: "THB", accounts: ["Joint Account"], symbol: "BRK-B")
        let target = try XCTUnwrap(ViewNavigation(url: url))
        XCTAssertEqual(target.section, .allocation)
        XCTAssertEqual(target.currency, "THB")
        XCTAssertEqual(target.accounts, ["Joint Account"])
        XCTAssertEqual(target.symbol, "BRK-B")
    }

    func testOriginalLayoutRestoresRedesignDefaultAndPreservesCustomization() {
        let redesign: Set<String> = ["portfolioHero", "performanceGraph", "annualTWR", "totalReturn", "cashBalance", "ledgerAllocation", "ledgerHoldings"]
        XCTAssertEqual(TabLayout.originalVisible(redesign), TabLayout.performanceInitialVisible)
        XCTAssertEqual(TabLayout.originalVisible(["cashBalance", "projection"]), ["cashBalance", "projection"])
        XCTAssertEqual(TabLayout.originalVisible(["cashBalance", "projection", "ledgerAllocation"]), ["cashBalance", "projection"])
        XCTAssertEqual(TabLayout.originalVisible(["ledgerHoldings"]), TabLayout.performanceInitialVisible)
    }

    private func settled(_ model: DashboardViewModel) async {
        for _ in 0..<200 {
            if !model.isLoading { return }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTFail("Dashboard did not finish")
    }
}
