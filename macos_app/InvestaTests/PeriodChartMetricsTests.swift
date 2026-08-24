import XCTest
@testable import Investa

/// `PeriodChartMetrics` decides how three charts draw themselves, and its
/// inputs are widths nobody can see in a screenshot review. These pin the two
/// decisions that were previously constants.
final class PeriodChartMetricsTests: XCTestCase {

    private func metrics(
        width: CGFloat,
        periods: Int,
        series: Int = 1,
        period: StatementPeriod = .quarterly
    ) -> PeriodChartMetrics {
        PeriodChartMetrics(
            containerWidth: width, periodCount: periods, seriesCount: series, periodType: period
        )
    }

    // MARK: Label capacity

    func testPhoneCarriesFewerLabelsThanDesktop() {
        // The card in the bug report: a 393pt phone, 16pt page + 16pt card padding.
        let phone = metrics(width: 329, periods: 20)
        let desktop = metrics(width: 900, periods: 20)
        XCTAssertEqual(phone.labelCapacity, 3)
        XCTAssertEqual(desktop.labelCapacity, 6, "Six is the cap; a wider frame doesn't earn more")
        XCTAssertLessThan(phone.labelCapacity, desktop.labelCapacity)
    }

    func testYearLabelsAreNarrowerSoMoreFit() {
        let quarterly = metrics(width: 329, periods: 20, period: .quarterly)
        let annual = metrics(width: 329, periods: 20, period: .annual)
        XCTAssertGreaterThan(annual.labelCapacity, quarterly.labelCapacity)
    }

    func testUnmeasuredWidthOpensOnThePhoneCount() {
        // The first frame must never be the overprinted one.
        XCTAssertEqual(metrics(width: 0, periods: 20).labelCapacity, 4)
    }

    func testCapacityNeverDropsBelowTwo() {
        XCTAssertEqual(metrics(width: 95, periods: 40).labelCapacity, 2)
    }

    // MARK: Thinning

    func testThinningKeepsTheNewestPeriod() {
        // The right-hand end is the label a reader looks for first.
        let periods = (1...20).map(String.init)
        let kept = metrics(width: 329, periods: 20).thinned(periods)
        XCTAssertEqual(kept.last, "20")
        XCTAssertLessThanOrEqual(kept.count, 3)
    }

    func testThinningIsAPassthroughWhenEverythingFits() {
        let periods = ["a", "b", "c"]
        XCTAssertEqual(metrics(width: 900, periods: 3).thinned(periods), periods)
    }

    // MARK: Bars vs. lines

    func testTwoSeriesOfQuartersStayBarsOnAPhone() {
        XCTAssertFalse(metrics(width: 329, periods: 20, series: 2).preferLines)
    }

    func testFourSeriesOfQuartersBecomeLinesOnAPhone() {
        // 239pt of plot over 20 periods across 4 series is 3pt of ink each.
        XCTAssertTrue(metrics(width: 329, periods: 20, series: 4).preferLines)
    }

    func testFourSeriesStayBarsWhenThereIsRoom() {
        XCTAssertFalse(metrics(width: 1200, periods: 20, series: 4).preferLines)
    }

    func testLongHistoriesAreLinesAtAnyWidth() {
        // Fifteen years of quarters is a shape, not a set of magnitudes.
        XCTAssertTrue(metrics(width: 2000, periods: 60).preferLines)
    }

    func testUnmeasuredWidthDoesNotFlipAShortHistoryToLines() {
        XCTAssertFalse(metrics(width: 0, periods: 20, series: 2).preferLines)
    }

    // MARK: Y domain

    func testDomainDoesNotGiveHalfTheFrameToOneSmallLoss() {
        // The reported card: twenty quarters of ~25B revenue with one -1B quarter.
        let domain = periodChartDomain([25_000_000_000, 24_000_000_000, -1_000_000_000])
        let range = try? XCTUnwrap(domain)
        guard let range else { return }
        XCTAssertGreaterThan(range.lowerBound, -5_000_000_000,
                             "A -20B floor is what Swift Charts' automatic domain rounds to")
        // The tallest bar should fill most of the frame, not 42% of it.
        let fill = 25_000_000_000 / (range.upperBound - range.lowerBound)
        XCTAssertGreaterThan(fill, 0.75)
    }

    func testAllPositiveDataSitsOnAZeroFloor() {
        let domain = periodChartDomain([10, 20, 30])
        XCTAssertEqual(domain?.lowerBound, 0)
        XCTAssertGreaterThan(domain?.upperBound ?? 0, 30)
    }

    func testAllNegativeDataKeepsZeroAtTheTop() {
        // Capital expenditure is reported negative; the axis is its ceiling.
        let domain = periodChartDomain([-10, -20, -30])
        XCTAssertEqual(domain?.upperBound, 0)
        XCTAssertLessThan(domain?.lowerBound ?? 0, -30)
    }

    // MARK: Stack decision

    func testUnmeasuredWidthPicksTheStackedShape() {
        // Rendering the wide shape first makes it demand its own minimum width,
        // which a vertical ScrollView adopts and re-proposes — after which the
        // probe measures the inflated width and the layout never narrows again.
        XCTAssertTrue(prefersStackedLayout(measuredWidth: 0, needs: 380))
        XCTAssertTrue(prefersStackedLayout(measuredWidth: -1, needs: 380))
    }

    func testStacksBelowWhatTheWideShapeNeeds() {
        XCTAssertTrue(prefersStackedLayout(measuredWidth: 329, needs: 380))
        XCTAssertFalse(prefersStackedLayout(measuredWidth: 380, needs: 380))
        XCTAssertFalse(prefersStackedLayout(measuredWidth: 900, needs: 380))
    }

    func testNoDomainWhereThereIsNothingToScale() {
        XCTAssertNil(periodChartDomain([]))
        XCTAssertNil(periodChartDomain([0, 0, 0]), "Swift Charts' own answer is better than 0...0")
    }
}
