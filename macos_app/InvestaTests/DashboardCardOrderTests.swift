import XCTest
@testable import Investa

/// The Dashboard's metric cards run in the web's order. The native grid once
/// kept its own hand-written list (Cash Balance and Total TWR first), so the
/// same portfolio read in a different order on the phone than in the browser.
final class DashboardCardOrderTests: XCTestCase {

    /// The web's `DEFAULT_ITEMS` order for the compact metric grid.
    private let webOrder = [
        "totalReturn", "unrealizedGL", "realizedGain", "annualTWR", "mwr",
        "ytdDividends", "dividendYield", "ytdReturn", "cashBalance",
        "fxGL", "fees", "taxes",
    ]

    func testMetricCardsFollowTheWebOrder() {
        XCTAssertEqual(TabLayout.metricCardOrder, webOrder)
    }

    /// Reads the web's own constants, so a reorder on either side fails here
    /// rather than on someone's phone. Skipped where the checkout isn't
    /// readable (a device run).
    func testWebConstantsStillListTheCardsInThisOrder() throws {
        let source = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("web_app/lib/dashboard_constants.ts")
        guard let text = try? String(contentsOf: source, encoding: .utf8) else {
            throw XCTSkip("web_app/lib/dashboard_constants.ts is not readable from this test run")
        }
        let start = try XCTUnwrap(text.range(of: "export const DEFAULT_ITEMS"))
        let body = text[start.upperBound...]
        let end = try XCTUnwrap(body.range(of: "];"))
        let regex = try NSRegularExpression(pattern: #"id:\s*'([A-Za-z]+)'"#)
        let block = String(body[..<end.lowerBound])
        let ids = regex.matches(in: block, range: NSRange(block.startIndex..., in: block)).compactMap {
            Range($0.range(at: 1), in: block).map { String(block[$0]) }
        }
        XCTAssertEqual(ids.filter { TabLayout.metricCardIds.contains($0) }, webOrder)
        XCTAssertEqual(TabLayout.performanceItems.map(\.id), ids,
                       "TabLayout.performanceItems must mirror the web's DEFAULT_ITEMS")
    }
}
