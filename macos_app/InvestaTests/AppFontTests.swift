import XCTest
import SwiftUI
@testable import Investa

/// macOS has no Dynamic Type, so nothing about the app's type size is enforced
/// by the system — `.dynamicTypeSize(.xLarge)` compiles there and changes
/// nothing, which is how the app spent a release rendering 10pt captions on a
/// 27" display. These pin the replacement: the scale curve, and the fact that a
/// scaled `AppFont` really does draw bigger glyphs than an unscaled one.
@MainActor
final class AppFontTests: XCTestCase {

    // MARK: - The curve

    func testWindowAtTheAppsMinimumWidthKeepsTodaysSizes() {
        XCTAssertEqual(AppFontScale.forWindowWidth(AppFontScale.referenceWidth), 1)
        // Before the first layout pass the measured width is 0. That must read
        // as "no bump", never as a divide-by-zero or a negative scale.
        XCTAssertEqual(AppFontScale.forWindowWidth(0), 1)
        XCTAssertEqual(AppFontScale.forWindowWidth(600), 1)
    }

    func testTypeGrowsWithTheWindow() {
        let midway = AppFontScale.forWindowWidth(
            (AppFontScale.referenceWidth + AppFontScale.fullScaleWidth) / 2)
        XCTAssertGreaterThan(midway, 1)
        XCTAssertLessThan(midway, AppFontScale.maximum)

        // Monotonic across the whole range a window can occupy.
        var previous = AppFontScale.forWindowWidth(0)
        for width in stride(from: CGFloat(700), through: 3000, by: 25) {
            let scale = AppFontScale.forWindowWidth(width)
            XCTAssertGreaterThanOrEqual(scale, previous, "scale shrank at \(width)pt")
            previous = scale
        }
    }

    func testTheBumpStopsSoAWideWindowIsNotAPoster() {
        XCTAssertEqual(AppFontScale.forWindowWidth(AppFontScale.fullScaleWidth), AppFontScale.maximum)
        XCTAssertEqual(AppFontScale.forWindowWidth(5120), AppFontScale.maximum)
    }

    func testScaleIsQuantisedSoADragDoesNotRelayoutEveryPixel() {
        // 5% steps: every value the curve can return is a multiple of 0.05.
        for width in stride(from: CGFloat(900), through: 1900, by: 7) {
            let steps = AppFontScale.forWindowWidth(width) * 20
            XCTAssertEqual(steps, steps.rounded(), accuracy: 1e-9, "\(width)pt is off-step")
        }
    }

    // MARK: - Resolution

    func testAnUnscaledFontIsTheOneTheCallSiteNames() {
        // Scale 1 must be a no-op, or every small window and every phone would
        // silently trade text-style leading (and iOS Dynamic Type) for a fixed
        // point size the moment the app adopted `.appFont(_:)`.
        assertRendersIdentically(AppFont.caption.resolved(scale: 1), .caption)
        assertRendersIdentically(AppFont.caption2.resolved(scale: 1), .caption2)
        assertRendersIdentically(AppFont.body.resolved(scale: 1), .body)
        assertRendersIdentically(AppFont.headline.resolved(scale: 1), .headline)
        assertRendersIdentically(AppFont.title2.resolved(scale: 1), .title2)
        assertRendersIdentically(AppFont.system(size: 11).resolved(scale: 1), .system(size: 11))
        assertRendersIdentically(AppFont.caption.weight(.bold).resolved(scale: 1),
                                 Font.caption.weight(.bold))
    }

    func testSemanticFontsActuallyGrow() {
        // The whole point: `.caption` is 10pt on macOS whatever the display.
        for font in [AppFont.caption, .caption2, .footnote, .subheadline, .body, .callout, .title3] {
            let plain = width(of: font.resolved(scale: 1))
            let scaled = width(of: font.resolved(scale: 1.3))
            XCTAssertGreaterThan(scaled, plain * 1.2, "\(font) did not grow with the window")
        }
    }

    func testFixedPointSizesGrowByTheSameProportion() {
        // Badges and tickers are written as `.system(size: 9…13)`. They have to
        // keep pace, or a scaled row's label outgrows the number beside it.
        let plain = width(of: AppFont.system(size: 10, weight: .bold).resolved(scale: 1))
        let scaled = width(of: AppFont.system(size: 10, weight: .bold).resolved(scale: 1.3))
        XCTAssertEqual(scaled / plain, 1.3, accuracy: 0.08)
    }

    func testScalingKeepsTheWeightTheStyleShipsWith() {
        // `.headline` is the one semibold text style; resolving it to an
        // explicit point size must not quietly flatten it to regular.
        let scaledHeadline = width(of: AppFont.headline.resolved(scale: 1.3))
        let scaledBody = width(of: AppFont.body.resolved(scale: 1.3))
        XCTAssertGreaterThan(scaledHeadline, scaledBody,
                             "scaled .headline lost its heavier weight")
    }

    func testScalingKeepsAnExplicitWeightAndDesign() {
        let regular = width(of: AppFont.caption.resolved(scale: 1.3))
        let black = width(of: AppFont.caption.weight(.black).resolved(scale: 1.3))
        XCTAssertGreaterThan(black, regular)

        // Monospaced digits are how numeric columns line up; the trim has to
        // survive the scale.
        let proportional = width(of: AppFont.body.resolved(scale: 1.3), text: "1111")
        let mono = width(of: AppFont.body.monospaced().resolved(scale: 1.3), text: "1111")
        XCTAssertNotEqual(proportional, mono, accuracy: 0.001)
    }

    // MARK: - Helpers

    /// Rendered width of a string at a given font, in points.
    private func width(of font: Font, text: String = "Hg 0123") -> CGFloat {
        ImageRenderer(content: Text(text).font(font).fixedSize()).nsImage?.size.width ?? 0
    }

    private func assertRendersIdentically(_ resolved: Font, _ expected: Font,
                                          file: StaticString = #filePath, line: UInt = #line) {
        let a = ImageRenderer(content: Text("Hg 0123").font(resolved).fixedSize()).nsImage?.size
        let b = ImageRenderer(content: Text("Hg 0123").font(expected).fixedSize()).nsImage?.size
        XCTAssertNotNil(a, file: file, line: line)
        XCTAssertEqual(a, b, file: file, line: line)
    }
}
