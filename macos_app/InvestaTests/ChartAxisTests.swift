import XCTest
import SwiftUI
@testable import Investa

/// `ChartAxis` decides how many labels the continuous charts draw, and its
/// inputs are widths and type sizes nobody can see in a screenshot review.
final class ChartAxisTests: XCTestCase {

    // MARK: Capacity

    /// The card in the bug reports: a 402pt phone, 16pt page padding each side.
    private let phoneCard: CGFloat = 370

    /// The phone's `.caption2` is 11pt; the macOS host these tests run on
    /// resolves 10. Scaling the host's own base up to the phone's puts the
    /// frame under test back on the device the screenshots came from.
    private var phoneType: CGFloat { 11 / ChartAxis.captionPointSize }

    func testPhoneCarriesFewerLabelsThanDesktop() {
        let label = ChartAxis.labelWidth("Dec 2010")
        let phone = ChartAxis.labelCapacity(width: phoneCard, labelWidth: label)
        let desktop = ChartAxis.labelCapacity(width: 1000, labelWidth: label)
        XCTAssertLessThan(phone, desktop)
        XCTAssertLessThanOrEqual(phone, 5, "Eight of these overprinted into a smear")
        XCTAssertEqual(desktop, 6, "Six is the cap; a wider frame doesn't earn more")
    }

    // MARK: Continuous axes

    func testAContinuousAxisCarriesFewerLabelsThanABandOne() {
        // The end labels hang inward from the plot edges, so each costs its
        // neighbour half a label of clearance that a band axis doesn't pay.
        let sample = "Dec 2025"
        XCTAssertLessThan(
            ChartAxis.tickCapacity(sample, width: phoneCard, scale: phoneType),
            ChartAxis.labelCapacity(
                width: phoneCard,
                labelWidth: ChartAxis.labelWidth(sample, scale: phoneType)
            )
        )
    }

    func testTheOneYearPositionHistoryFitsItsDates() {
        // The reported frame: four "Dec 2025" labels touched, two per end, with
        // the middle of the axis standing empty.
        XCTAssertEqual(ChartAxis.tickCapacity("Dec 2025", width: phoneCard, scale: phoneType), 3)
    }

    func testShorterDatesStillEarnMoreTicks() {
        XCTAssertGreaterThan(
            ChartAxis.tickCapacity("30 Sep", width: phoneCard, scale: phoneType),
            ChartAxis.tickCapacity("Dec 2025", width: phoneCard, scale: phoneType)
        )
    }

    func testAWideChartStillCapsAtSix() {
        XCTAssertEqual(ChartAxis.tickCapacity("Dec 2025", width: 1400), 6)
    }

    func testTicksNeverDropBelowTwoOrOpenWide() {
        XCTAssertEqual(ChartAxis.tickCapacity("Dec 2025", width: 100), 2)
        XCTAssertEqual(ChartAxis.tickCapacity("Dec 2025", width: 0), 3,
                       "Unmeasured opens on the phone count, never the crowded one")
    }

    func testAccessibilityTypeEarnsFewerTicks() {
        XCTAssertLessThan(
            ChartAxis.tickCapacity("Dec 2025", width: phoneCard, typeSize: .accessibility3),
            ChartAxis.tickCapacity("Dec 2025", width: phoneCard)
        )
    }

    func testUnmeasuredWidthOpensOnThePhoneCount() {
        // The first frame must never be the overprinted one.
        XCTAssertEqual(ChartAxis.labelCapacity(width: 0, labelWidth: 60), 3)
    }

    func testCapacityNeverDropsBelowTwo() {
        XCTAssertEqual(ChartAxis.labelCapacity(width: 90, labelWidth: 60), 2)
    }

    func testAccessibilityTypeEarnsFewerLabels() {
        let normal = ChartAxis.labelWidth("Dec 2010")
        let large = ChartAxis.labelWidth("Dec 2010", typeSize: .accessibility3)
        XCTAssertGreaterThan(large, normal)
        XCTAssertLessThan(
            ChartAxis.labelCapacity(width: 370, labelWidth: large),
            ChartAxis.labelCapacity(width: 370, labelWidth: normal),
            "A label ~1.9x wider cannot fit the same count"
        )
    }

    func testShorterLabelsEarnMore() {
        XCTAssertGreaterThan(
            ChartAxis.labelCapacity(width: 370, labelWidth: ChartAxis.labelWidth("Sep")),
            ChartAxis.labelCapacity(width: 370, labelWidth: ChartAxis.labelWidth("Dec 2010"))
        )
    }

    // MARK: Ticks

    func testTicksKeepBothEnds() {
        // "Since when" and "as of when" are the two labels a reader looks for.
        let days = Array(1...3650)
        let ticks = ChartAxis.ticks(days, count: 4)
        XCTAssertEqual(ticks.first, 1)
        XCTAssertEqual(ticks.last, 3650)
        XCTAssertEqual(ticks.count, 4)
    }

    func testTicksAreEvenlySpaced() {
        let ticks = ChartAxis.ticks(Array(0...100), count: 5)
        XCTAssertEqual(ticks, [0, 25, 50, 75, 100])
    }

    func testTicksArePassthroughWhenEverythingFits() {
        let months = ["Jan", "Feb", "Mar"]
        XCTAssertEqual(ChartAxis.ticks(months, count: 6), months)
    }

    func testTicksNeverRepeatAValue() {
        // A series can repeat an x value; two marks on one tick draw twice.
        let flat = Array(repeating: "Mar", count: 20)
        XCTAssertEqual(ChartAxis.ticks(flat, count: 5), ["Mar"])
    }

    func testTicksHandleAnEmptySeries() {
        XCTAssertTrue(ChartAxis.ticks([Int](), count: 5).isEmpty)
    }

    // MARK: Month initials

    func testTwelveMonthsUseInitialsOnAPhone() {
        // `Jan Feb Mar` twelve deep wants ~310pt; the card has ~310 all in.
        XCTAssertTrue(ChartAxis.prefersMonthInitials(count: 12, width: 370))
    }

    func testTwelveMonthsKeepTheirNamesWhereThereIsRoom() {
        XCTAssertFalse(ChartAxis.prefersMonthInitials(count: 12, width: 1000))
    }

    func testInitialsAreNotUsedWhenTheyWouldStillBeThinned() {
        // A broken run of initials can't be counted back to a month; a thinned
        // "Mar" still names itself.
        XCTAssertFalse(ChartAxis.prefersMonthInitials(count: 12, width: 150))
        XCTAssertFalse(ChartAxis.prefersMonthInitials(count: 12, width: 370,
                                                      typeSize: .accessibility5))
    }

    func testUnmeasuredWidthKeepsTheMonthNames() {
        // The first frame is the safe one: names, thinned.
        XCTAssertFalse(ChartAxis.prefersMonthInitials(count: 12, width: 0))
    }

    // MARK: Anchors

    func testEdgeLabelsHangInward() {
        // Centred on the outermost tick, a label hangs half its width past the
        // plot and is clipped there.
        let ticks = [1, 2, 3]
        XCTAssertEqual(ChartAxis.anchor(1, in: ticks), .topLeading)
        XCTAssertEqual(ChartAxis.anchor(2, in: ticks), .top)
        XCTAssertEqual(ChartAxis.anchor(3, in: ticks), .topTrailing)
    }

    func testALoneLabelIsCentred() {
        XCTAssertEqual(ChartAxis.anchor(1, in: [1]), .top)
    }
}
