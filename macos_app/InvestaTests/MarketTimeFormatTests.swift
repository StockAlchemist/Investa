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

    func testMonthInitialIsTheInitialAndNotACutAbbreviation() {
        XCTAssertEqual(MarketTime.monthInitial("2026-09-01"), "S")
        XCTAssertEqual(MarketTime.monthInitial("2026-06-15"), "J")
        // The whole run, which is what a narrow axis draws.
        let months = (1...12).map { MarketTime.monthInitial(String(format: "2026-%02d-01", $0)) }
        XCTAssertEqual(months, ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    }

    func testMonthInitialAcceptsABucketKey() {
        // `projected_income` keys its months "yyyy-MM", with no day to parse.
        XCTAssertEqual(MarketTime.monthInitial("2026-09"), "S")
    }

    func testMonthInitialLeavesNonDatesAlone() {
        XCTAssertEqual(MarketTime.monthInitial("Portfolio"), "Portfolio")
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

    // MARK: The year is the market's, not the device's

    func testTheDisplayLocaleDropsADeviceEraCalendar() {
        // A device set to a Thai region reckons in the Buddhist era, which would
        // date a trade filed in 2018 to 2561.
        let thai = MarketTime.gregorianLocale(Locale(identifier: "en_TH"))
        XCTAssertEqual(thai.calendar.identifier, .gregorian)
        XCTAssertEqual(MarketTime.displayLocale.calendar.identifier, .gregorian)
    }

    func testTheDisplayLocaleKeepsTheReaderSLanguage() {
        // Only the calendar is swapped: month names still come from the reader's
        // own locale, which is the half of the notation that stays localized.
        let thai = MarketTime.gregorianLocale(Locale(identifier: "en_TH"))
        XCTAssertEqual(thai.language.languageCode, Locale.Language(identifier: "en").languageCode)
        XCTAssertEqual(thai.region, Locale.Region("TH"))
    }

    func testAFormatterNamesTheGregorianYearUnderAnEraLocale() {
        // The chart tooltip that reported this: "Thu, 23 Aug 2561" for a day
        // every exchange calls 2018.
        let f = MarketTime.formatter("EEE, dd MMM yyyy", timeZone: TimeZone(identifier: "UTC"))
        f.locale = MarketTime.gregorianLocale(Locale(identifier: "en_TH"))
        let day = try? XCTUnwrap(MarketTime.calendarDay("2018-08-23"))
        guard let day else { return }
        let rendered = f.string(from: day)
        XCTAssertTrue(rendered.hasSuffix("2018"), rendered)
        XCTAssertFalse(rendered.contains("2561"), rendered)
    }

    func testTheWireFormatterWritesAGregorianISODay() {
        // This one is not a notation but a payload: a Buddhist year here posts
        // "2561-08-23" to the backend and stores a trade 543 years out.
        let f = MarketTime.isoFormatter(timeZone: TimeZone(identifier: "UTC"))
        let day = try? XCTUnwrap(MarketTime.calendarDay("2018-08-23"))
        guard let day else { return }
        XCTAssertEqual(f.string(from: day), "2018-08-23")
        XCTAssertEqual(f.calendar.identifier, .gregorian)
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
