import SwiftUI

/// Responsive KPI container shared by every KPI strip.
///
/// Lays the tiles out in a grid whose column count is derived from the *actual*
/// available width (not the platform or size class — an iPad's size class stays
/// `.regular` whether the sidebar is open, closed, or it's in landscape). The
/// tiles are then balanced evenly across rows, so there's never a lone tile
/// stranded on the last row and never a single row so tight that values
/// truncate. When everything fits comfortably in one row (wide iPad / macOS),
/// that's exactly what you get.
struct KpiRow<Content: View>: View {
    /// Number of tiles in `content` — used to balance them across rows.
    var count: Int
    /// Comfortable minimum tile width; drives how many columns fit.
    var minTileWidth: CGFloat = 150
    @ViewBuilder var content: Content

    @State private var availableWidth: CGFloat = 0

    private var columnCount: Int {
        guard count > 1 else { return max(1, count) }
        // Until the first measurement lands, assume a single row.
        guard availableWidth > 0 else { return count }
        // A phone in Display Zoom, or at an accessibility text size, measures
        // narrower than two comfortable tiles — but one tile per row is a list,
        // not a strip: it leaves half the card empty and pushes everything below
        // it off the screen. Keep two columns wherever two tiles still fit
        // compressed; every tile's value shrinks before it truncates.
        let floorCols = availableWidth >= minTileWidth * 1.2 ? 2 : 1
        let maxCols = max(floorCols, Int(availableWidth / minTileWidth))
        let perRowCap = min(maxCols, count)
        // Balance: spread tiles over the fewest rows `maxCols` allows, then even
        // them out so the last row is as full as the rest (e.g. 7 tiles, 5 cols
        // → 2 rows of 4+3 rather than 5+2).
        let rows = Int(ceil(Double(count) / Double(perRowCap)))
        return Int(ceil(Double(count) / Double(max(1, rows))))
    }

    var body: some View {
        let columns = Array(repeating: GridItem(.flexible(), spacing: 12), count: columnCount)
        LazyVGrid(columns: columns, alignment: .leading, spacing: 12) {
            content
        }
        .frame(maxWidth: .infinity)
        // Must be the *offered* width: measuring the resolved width latches a
        // wide column count on after a landscape trip. See `readingContainerWidth`.
        .readingContainerWidth { availableWidth = $0 }
    }
}
