import XCTest
@testable import Investa

/// The hero chart's hover card names the moment under the cursor. The failure
/// is silent and time-zone shaped: a zoneless intraday timestamp read on the
/// device clock puts the New York open at 02:30 for a reader in Bangkok.
final class HeroChartTooltipTests: XCTestCase {

    func testZonelessIntradayTimestampIsUTCShownOnTheMarketClock() {
        // 14:30 UTC is 10:30 in New York (EDT) on 24 Sep 2026.
        XCTAssertEqual(PortfolioHeroCard.pointLabel("2026-09-24 14:30:00", weekday: true), "Thu, 24 Sep 10:30 AM")
    }

    func testOffsetTimestampKeepsItsOwnZone() {
        XCTAssertEqual(PortfolioHeroCard.pointLabel("2026-09-24T10:30:00-04:00", weekday: true), "Thu, 24 Sep 10:30 AM")
        XCTAssertEqual(PortfolioHeroCard.pointLabel("2026-09-24T14:30:00Z", weekday: false), "24 Sep 10:30 AM")
    }

    func testDailyPointIsACalendarDay() {
        XCTAssertEqual(PortfolioHeroCard.pointLabel("2026-09-24", weekday: true), "24 Sep 2026")
    }
}
