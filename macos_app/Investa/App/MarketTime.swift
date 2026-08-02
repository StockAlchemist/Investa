import Foundation

/// Market-local date reckoning — the SwiftUI twin of `web_app/lib/market_time.ts`
/// and `src/server/calendar_events.py`.
///
/// Every calendar date Investa shows belongs to a market, not to the device: an
/// earnings report is "today" while it is still today on that exchange, whether
/// the phone is in Bangkok or Los Angeles. Counting days against `Calendar.current`
/// instead is what put a US report a day out for a viewer in Asia.
enum MarketTime {
    /// Fallback zone for events that do not name their exchange's (Investa is US-first).
    static let defaultTimeZoneIdentifier = "America/New_York"

    private static let utc = TimeZone(identifier: "UTC") ?? TimeZone(secondsFromGMT: 0)!

    /// Calendar days are compared in UTC so that no exchange's DST transition can
    /// land between the two dates being subtracted.
    private static let utcCalendar: Calendar = {
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = utc
        return cal
    }()

    private static let dayParser: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd"
        f.timeZone = utc
        return f
    }()

    /// A market date belongs to the market's calendar, not the device's. A
    /// phone set to Thailand defaults to the Buddhist era and renders a fiscal
    /// quarter ending March 2026 as "Mar 2569 BE" — the same class of mistake
    /// as reading a US market date on a Bangkok clock, and just as wrong on a
    /// filed statement. Pinning the calendar keeps the year the filer's own
    /// while leaving month names localized.
    private static let gregorian = Calendar(identifier: .gregorian)

    /// "Jul 29, 2026" — a calendar day, so it renders in UTC to match the parser.
    private static let mediumFormatter: DateFormatter = {
        let f = DateFormatter()
        f.calendar = gregorian
        f.dateStyle = .medium
        f.timeStyle = .none
        f.timeZone = utc
        return f
    }()

    /// "Jul 29" — the compact form for rows too narrow for a full date.
    private static let shortDayFormatter: DateFormatter = {
        let f = DateFormatter()
        f.calendar = gregorian
        f.setLocalizedDateFormatFromTemplate("MMMd")
        f.timeZone = utc
        return f
    }()

    /// "Mar 2026" — how a quarter is named, since four columns a year would
    /// otherwise all read "2026".
    ///
    /// Fixed English rather than localized, matching `yearFormatter` below: a
    /// localized month-year template carries an era marker in some regions
    /// ("Mar 2026 AD" for en_TH), and an axis label has no room for it.
    private static let monthYearFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.calendar = gregorian
        f.dateFormat = "MMM yyyy"
        f.timeZone = utc
        return f
    }()

    /// "2026" — the axis label for a chart plotted on calendar days.
    private static let yearFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy"
        f.timeZone = utc
        return f
    }()

    /// The calendar day an ISO date (or datetime) string names, at UTC midnight.
    static func calendarDay(_ iso: String) -> Date? {
        dayParser.date(from: String(iso.prefix(10)))
    }

    /// The exchange's zone, falling back to the US default for a missing or
    /// unrecognized identifier.
    static func zone(_ identifier: String?) -> TimeZone {
        if let trimmed = identifier?.trimmingCharacters(in: .whitespaces), !trimmed.isEmpty,
           let tz = TimeZone(identifier: trimmed) {
            return tz
        }
        return TimeZone(identifier: defaultTimeZoneIdentifier) ?? utc
    }

    /// Today's calendar day on a market's own clock, at UTC midnight so it can be
    /// differenced against `calendarDay(_:)`.
    static func today(timeZone identifier: String?) -> Date? {
        var marketCalendar = Calendar(identifier: .gregorian)
        marketCalendar.timeZone = zone(identifier)
        let parts = marketCalendar.dateComponents([.year, .month, .day], from: Date())
        return utcCalendar.date(from: DateComponents(year: parts.year, month: parts.month, day: parts.day))
    }

    /// Whole days from today-on-the-market to `iso`. Negative for the past, nil if
    /// the date can't be read.
    static func dayDiff(_ iso: String, timeZone identifier: String?) -> Int? {
        guard let target = calendarDay(iso), let today = today(timeZone: identifier) else { return nil }
        return utcCalendar.dateComponents([.day], from: today, to: target).day
    }

    /// Whether a calendar date falls no later than `months` months past today on a
    /// market's own clock — the horizon behind the "3 Months / 1 Year" calendar
    /// toggles. A month is a calendar month, clamped to the month's last day.
    static func isWithin(_ iso: String, months: Int, timeZone identifier: String?) -> Bool {
        guard let target = calendarDay(iso), let today = today(timeZone: identifier),
              let cutoff = utcCalendar.date(byAdding: .month, value: months, to: today)
        else { return false }
        return target <= cutoff
    }

    /// "Jul 29, 2026" for a date-only string; the input is returned unchanged if
    /// it isn't a date.
    static func formatted(_ iso: String) -> String {
        guard let d = calendarDay(iso) else { return iso }
        return mediumFormatter.string(from: d)
    }

    /// "Jul 29" for a date-only string.
    static func shortDay(_ iso: String) -> String {
        guard let d = calendarDay(iso) else { return iso }
        return shortDayFormatter.string(from: d)
    }

    /// "Jul 2026" for a date-only string.
    static func monthYear(_ iso: String) -> String {
        guard let d = calendarDay(iso) else { return iso }
        return monthYearFormatter.string(from: d)
    }

    /// "Jul 2026" for a day already parsed by `calendarDay(_:)` — the form a
    /// chart axis hands back.
    static func monthYear(_ day: Date) -> String {
        monthYearFormatter.string(from: day)
    }

    /// The calendar year of a day produced by `calendarDay(_:)`, read back in the
    /// same zone it was built in.
    static func year(_ day: Date) -> String {
        yearFormatter.string(from: day)
    }
}
