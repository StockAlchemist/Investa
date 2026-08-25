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

    /// The reader's own locale with the Gregorian calendar forced on.
    ///
    /// A device set to a Thai region defaults to the Buddhist era, so a
    /// formatter that inherits it renders 23 August 2018 as "23 Aug 2561" — a
    /// year no market, filing or trade confirmation has ever used. Setting only
    /// `DateFormatter.calendar` is not enough on its own: a formatter takes its
    /// calendar from its locale, so the calendar has to be baked into the
    /// locale. Month and weekday names stay in the reader's language; only the
    /// era changes.
    static let displayLocale: Locale = gregorianLocale(.current)

    /// `base` with the Gregorian calendar substituted, and nothing else touched.
    static func gregorianLocale(_ base: Locale) -> Locale {
        if base.calendar.identifier == .gregorian { return base }
        var components = Locale.Components(locale: base)
        components.calendar = .gregorian
        return Locale(components: components)
    }

    /// The Gregorian calendar on the device's own zone — for arithmetic on dates
    /// the reader picked ("a year ago", "the start of this year").
    /// `Calendar.current` is the calendar the device *displays*, so on a Thai
    /// device `component(.year:)` answers 2569 and any comparison against a year
    /// parsed from the API is off by 543.
    static var localCalendar: Calendar {
        var cal = gregorian
        cal.timeZone = .current
        return cal
    }

    /// The US market's zone — the default for an axis or tooltip plotting a
    /// series that does not name its own exchange.
    static var defaultZone: TimeZone { zone(nil) }

    /// A display formatter pinned to the Gregorian calendar, and to a market's
    /// zone when one is given. Every user-visible date the app builds outside
    /// this file goes through here: a bare `DateFormatter()` inherits the
    /// device's calendar and prints the wrong year.
    static func formatter(_ pattern: String, timeZone: TimeZone? = nil) -> DateFormatter {
        let f = DateFormatter()
        // Locale first — assigning one resets `calendar` to the locale's own.
        f.locale = displayLocale
        f.calendar = gregorian
        f.dateFormat = pattern
        if let timeZone { f.timeZone = timeZone }
        return f
    }

    /// A formatter for the ISO `yyyy-MM-dd` the API speaks — fixed English and
    /// Gregorian, since this is a wire format, not a notation a reader sees.
    /// A device calendar leaking in here would send the backend "2569-08-25".
    static func isoFormatter(timeZone: TimeZone? = nil) -> DateFormatter {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.calendar = gregorian
        f.dateFormat = "yyyy-MM-dd"
        if let timeZone { f.timeZone = timeZone }
        return f
    }

    /// "29 Jul 2026" — `DD MMM YYYY`, the app's one date notation.
    ///
    /// An explicit pattern rather than `dateStyle = .medium`, which orders the
    /// parts by the device's locale: the same lot would read "Jul 29, 2026" on
    /// a US phone and "29/07/2026" on a British one, and a list of dates that
    /// changes shape with the reader is not a format. Month names stay
    /// localized; only the order and the zero-padded day are fixed, so figures
    /// line up down a column.
    ///
    /// Renders in UTC to match the parser — a calendar day is a day, not an
    /// instant, and must not slide for a viewer west of UTC.
    private static let mediumFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = displayLocale
        f.calendar = gregorian
        f.dateFormat = "dd MMM yyyy"
        f.timeZone = utc
        return f
    }()

    /// "29 Jul" — the same notation with the year dropped, for a row or axis
    /// with no room for it. Day-first like the full form; never "Jul 29".
    private static let shortDayFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = displayLocale
        f.calendar = gregorian
        f.dateFormat = "dd MMM"
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
        f.calendar = gregorian
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

    /// "29 Jul 2026" for a date-only string; the input is returned unchanged if
    /// it isn't a date.
    static func formatted(_ iso: String) -> String {
        guard let d = calendarDay(iso) else { return iso }
        return mediumFormatter.string(from: d)
    }

    /// "29 Jul 2026" for a day already parsed by `calendarDay(_:)`. Only for
    /// calendar days: an intraday timestamp would render in UTC and can name
    /// the wrong day.
    static func formatted(_ day: Date) -> String {
        mediumFormatter.string(from: day)
    }

    /// "29 Jul" for a date-only string.
    static func shortDay(_ iso: String) -> String {
        guard let d = calendarDay(iso) else { return iso }
        return shortDayFormatter.string(from: d)
    }

    /// "29 Jul" for a day already parsed by `calendarDay(_:)`.
    static func shortDay(_ day: Date) -> String {
        shortDayFormatter.string(from: day)
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
