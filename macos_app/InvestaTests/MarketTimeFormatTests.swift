import XCTest
@testable import Investa

/// `DD MMM YYYY` is the app's one date notation. These pin it, because the
/// failure mode is silent: `dateStyle = .medium` compiles, looks right on the
/// developer's US-locale machine, and renders "Jul 29, 2026" or "29/07/2026"
/// depending on who is holding the phone.
final class MarketTimeFormatTests: XCTestCase {

    func testFullDateIsDayMonthYear() {
        XCTAssertEqual(MarketTime.formatted("2026-07-29"), "29 Jul 2026")
        XCTAssertEqual(MarketTime.formatted("2024-01-15"), "15 Jan 2024")
    }

    func testDayIsZeroPaddedSoColumnsAlign() {
        XCTAssertEqual(MarketTime.formatted("2026-03-05"), "05 Mar 2026")
    }

    func testDatetimeStringsUseTheirCalendarDay() {
        XCTAssertEqual(MarketTime.formatted("2026-07-29T14:30:00Z"), "29 Jul 2026")
    }

    func testShortFormKeepsTheDayFirstOrder() {
        // Dropping the year must not reorder what is left into "Jul 29".
        XCTAssertEqual(MarketTime.shortDay("2026-07-29"), "29 Jul")
        XCTAssertEqual(MarketTime.shortDay("2026-03-05"), "05 Mar")
    }

    func testMonthYearHasNoDayToPlace() {
        XCTAssertEqual(MarketTime.monthYear("2026-07-29"), "Jul 2026")
    }

    func testNonDatesPassThroughUnchanged() {
        // A label that isn't a date must not be mangled into one.
        XCTAssertEqual(MarketTime.formatted("—"), "—")
        XCTAssertEqual(MarketTime.formatted(""), "")
    }

    func testACalendarDayDoesNotSlideForViewersWestOfUTC() {
        // Date-only strings are days, not instants: rendered in the viewer's
        // zone, 2026-01-01 becomes 31 Dec for anyone west of UTC.
        let day = try? XCTUnwrap(MarketTime.calendarDay("2026-01-01"))
        guard let day else { return }
        XCTAssertEqual(MarketTime.formatted(day), "01 Jan 2026")
    }

    func testTransactionsShowTheNotationNotTheRawISOString() {
        let tx = Transaction(
            id: nil, date: "2024-01-15", account: "IBKR", symbol: "GOOG",
            type: "Buy", quantity: 10, pricePerShare: 100, commission: 0,
            totalAmount: 1000, localCurrency: "USD"
        )
        XCTAssertEqual(tx.displayDate, "15 Jan 2024")
    }
}
