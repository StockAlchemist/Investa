import XCTest
import SwiftUI
@testable import Investa

/// `KpiRow` decides how many columns every KPI strip in the app lays out, from
/// the width it was offered. The failure it exists to prevent — two tiles
/// squeezed side by side until a figure wraps mid-number — is invisible in a
/// screenshot taken on a wide window.
final class KpiRowTests: XCTestCase {

    /// A 402pt phone: 16pt page padding and 16pt card padding each side.
    private let phoneStrip: CGFloat = 338
    /// An iPad in portrait, same insets.
    private let padStrip: CGFloat = 746

    // MARK: Stat tiles — a label and a number, which shrink when squeezed

    func testStatTilesHoldTwoColumnsOnAPhone() {
        // One stat tile per row is a list, not a strip.
        XCTAssertEqual(
            KpiGrid.columnCount(count: 6, availableWidth: phoneStrip, minTileWidth: 140), 2
        )
    }

    func testStatTilesSpreadOutWhenThereIsRoom() {
        XCTAssertGreaterThan(
            KpiGrid.columnCount(count: 6, availableWidth: padStrip, minTileWidth: 140),
            KpiGrid.columnCount(count: 6, availableWidth: phoneStrip, minTileWidth: 140)
        )
    }

    func testRowsAreBalancedRatherThanLeavingAStrandedTile() {
        // 7 tiles across 5 columns is 4+3, not 5+2.
        XCTAssertEqual(
            KpiGrid.columnCount(count: 7, availableWidth: 750, minTileWidth: 140), 4
        )
    }

    // MARK: Composite cards — nothing left to give at half a phone's width

    func testACompositeCardTakesTheWholeRowOnAPhone() {
        // The transactions currency cards: a net figure, a currency badge and
        // two label/value rows. Two per row wrapped "+49.2K" into "+49." / "2K".
        XCTAssertEqual(
            KpiGrid.columnCount(count: 2, availableWidth: phoneStrip,
                               minTileWidth: 260, floorColumns: 1), 1
        )
    }

    func testTheOldBudgetIsWhatPutTwoOfThemOnAPhone() {
        // Pins the regression: the floor, not the tile width, forced the pair.
        XCTAssertEqual(
            KpiGrid.columnCount(count: 2, availableWidth: phoneStrip, minTileWidth: 240), 2
        )
    }

    func testCompositeCardsPairUpFromAnIPadInPortrait() {
        XCTAssertEqual(
            KpiGrid.columnCount(count: 2, availableWidth: padStrip,
                               minTileWidth: 260, floorColumns: 1), 2
        )
        XCTAssertEqual(
            KpiGrid.columnCount(count: 4, availableWidth: 1400,
                               minTileWidth: 260, floorColumns: 1), 4
        )
    }

    func testNoColumnFloorStillNeverReturnsZero() {
        XCTAssertEqual(
            KpiGrid.columnCount(count: 3, availableWidth: 80,
                               minTileWidth: 260, floorColumns: 1), 1
        )
    }

    // MARK: Degenerate inputs

    func testASingleTileIsOneColumn() {
        XCTAssertEqual(KpiGrid.columnCount(count: 1, availableWidth: 1400), 1)
        XCTAssertEqual(KpiGrid.columnCount(count: 0, availableWidth: 1400), 1)
    }

    func testUnmeasuredWidthAssumesOneRow() {
        XCTAssertEqual(KpiGrid.columnCount(count: 5, availableWidth: 0), 5)
    }
}
