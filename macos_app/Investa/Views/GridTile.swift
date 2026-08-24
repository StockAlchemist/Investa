import SwiftUI

extension View {
    /// Marks a card that sits in a row or grid alongside its peers.
    ///
    /// A grid row is only as tall as its tallest cell, and a cell that sizes to
    /// its own content leaves the rest of the row short — so one tile carrying a
    /// two-line label or a sub-value stands taller than the tile beside it and
    /// the row reads as broken rather than as a set. `maxHeight: .infinity`
    /// makes each tile take the height the row already has.
    ///
    /// Fills width too, since a tile that doesn't is the same defect on the
    /// other axis. Content stays pinned to the top-leading corner: a value
    /// centred in whatever height its neighbour happened to need would drift
    /// row by row.
    ///
    /// Only for cards laid out beside each other. A card alone in a `VStack`
    /// has no row height to match and would stretch to fill whatever it is
    /// offered.
    func gridTile(alignment: Alignment = .topLeading) -> some View {
        frame(maxWidth: .infinity, maxHeight: .infinity, alignment: alignment)
    }
}
