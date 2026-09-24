import SwiftUI

// MARK: - Ledger palette
//
// The same tokens as `web_app/app/globals.css` (`:root` / `.dark`), so a card,
// a gain and the accent are the same object on all three clients. Warm paper,
// white cards, near-black ink, one cobalt accent. Colour means something —
// gain, loss, caution, "you can act on this" — and is never decoration.

extension Color {
    /// Adaptive colour from two hex values (sRGB), light first.
    static func adaptive(lightHex: UInt, darkHex: UInt) -> Color {
        func rgb(_ h: UInt) -> (r: Double, g: Double, b: Double) {
            (Double((h >> 16) & 0xff) / 255, Double((h >> 8) & 0xff) / 255, Double(h & 0xff) / 255)
        }
        return adaptive(light: rgb(lightHex), dark: rgb(darkHex))
    }

    // Ground and ink.
    /// The page ground — warm paper (#F5F4EF) / near-black (#0F1013).
    static let paper = Color.adaptive(lightHex: 0xF5F4EF, darkHex: 0x0F1013)
    /// A panel inside a card: the inset tier's fill, and segmented-control tracks.
    static let inset = Color.adaptive(lightHex: 0xF1F0EA, darkHex: 0x1E2127)
    /// Hairline rule between rows and around cards.
    static let line = Color.adaptive(lightHex: 0xE4E2DA, darkHex: 0x2A2D34)
    /// The stronger rule used around form fields.
    static let lineStrong = Color.adaptive(lightHex: 0xD3D0C6, darkHex: 0x34373F)
    /// Second text tone: context under a figure ("p.a.", "on cost").
    static let ink2 = Color.adaptive(lightHex: 0x4C4E56, darkHex: 0xB3B5BC)
    /// Third text tone: labels and captions. ≥4.5:1 on paper and on a card.
    static let ink3 = Color.adaptive(lightHex: 0x6A6C74, darkHex: 0x8D9098)

    // Semantic. Gain and loss always travel with a sign, so hue is never the only cue.
    /// Gain.
    static let up = Color.adaptive(lightHex: 0x0B6B4C, darkHex: 0x4FC897)
    /// Loss.
    static let down = Color.adaptive(lightHex: 0xB83A15, darkHex: 0xFF8D6B)
    /// Caution — stale data, drift over target, FX. Legible as text on a card.
    static let warn = Color.adaptive(lightHex: 0x9A6200, darkHex: 0xF1BC55)
    static let upTint = Color.adaptive(lightHex: 0xE3F1EA, darkHex: 0x173127)
    static let downTint = Color.adaptive(lightHex: 0xFBE9E3, darkHex: 0x3A1F17)
    static let warnTint = Color.adaptive(lightHex: 0xFBF0D9, darkHex: 0x3A2E14)

    /// The one interface accent — cobalt. Primary buttons, the selected
    /// segment's ink, focus, links. Never a gain or a loss.
    static let brand = Color.adaptive(lightHex: 0x2E46C8, darkHex: 0x8FA3FF)
    /// Accent ink on a tinted fill.
    static let brandInk = Color.adaptive(lightHex: 0x2A3FB5, darkHex: 0xA9B8FF)
    /// Accent tint — the fill behind an active chip.
    static let brandTint = Color.adaptive(lightHex: 0xE8EBFA, darkHex: 0x232A4D)

    // Chart series — categorical, never chrome. Mid-tones legible on both grounds.
    static let plum = Color.adaptive(lightHex: 0x8B4FA6, darkHex: 0xB17FCB)
    static let dataTeal = Color.adaptive(lightHex: 0x1C8C84, darkHex: 0x4FB0A8)
    static let ochre = Color.adaptive(lightHex: 0xC8921E, darkHex: 0xF1BC55)
    static let periwinkle = Color.adaptive(lightHex: 0x5F79DB, darkHex: 0x7C93E8)
    static let graphite = Color.adaptive(lightHex: 0x6A6C74, darkHex: 0x9A9CA3)

    // Retired names, re-pointed at Ledger so existing call sites land on the
    // design language. New code names the role (`brand`, `plum`, `warn`).
    static let brandIndigo = brand
    static let brandTeal = dataTeal
    static let brandViolet = plum
    static let brandPurple = plum
    static let brandAmber = ochre
    static let brandCyan = dataTeal
    static let brandSky = periwinkle
    static let brandEmerald = Color.adaptive(lightHex: 0x1F9D6C, darkHex: 0x4FC897)
    static let brandRose = Color.adaptive(lightHex: 0xD2491F, darkHex: 0xFF8D6B)

    /// Overline and small label colour.
    static let sectionText = ink3

    /// The chosen segment of a segmented control: raised white on the inset
    /// track (light), the strong rule colour (dark). Its text is `.primary`,
    /// the others `.ink2` — the twin of the web `.segmented`.
    static let segmentOn = Color.adaptive(lightHex: 0xFFFFFF, darkHex: 0x34373F)

    /// Card surface: white on paper / #17191E on near-black.
    static let cardBg = Color.adaptive(lightHex: 0xFFFFFF, darkHex: 0x17191E)
    /// Card rule.
    static let cardBorder = line
}

// Make the Ledger colours usable as implicit members in ShapeStyle contexts
// (.foregroundStyle, .fill, .background, chart .foregroundStyle) — like the
// built-in `.red`/`.green`, which Ledger replaces.
extension ShapeStyle where Self == Color {
    static var up: Color { Color.up }
    static var down: Color { Color.down }
    static var warn: Color { Color.warn }
    static var brand: Color { Color.brand }
    static var brandIndigo: Color { Color.brandIndigo }
    static var sectionText: Color { Color.sectionText }
    static var paper: Color { Color.paper }
    static var inset: Color { Color.inset }
    static var line: Color { Color.line }
    static var ink2: Color { Color.ink2 }
    static var ink3: Color { Color.ink3 }
    static var plum: Color { Color.plum }
    static var dataTeal: Color { Color.dataTeal }
    static var ochre: Color { Color.ochre }
    static var periwinkle: Color { Color.periwinkle }
}

/// App-wide visual tokens. Centralizes the card chrome that was previously
/// copy-pasted across every feature, so the whole app can be retuned in one place.
enum Theme {
    /// The one interface accent.
    static let brand = Color.brand

    /// Categorical colours for charts and legends — fixed order, never used for
    /// chrome. A tab is not a colour; a sector slice is. Same order as the web
    /// allocation palette.
    static let dataPalette: [Color] = [
        Color.brand,       // cobalt
        Color.dataTeal,    // teal
        Color.ochre,       // ochre
        Color.plum,        // plum
        Color.periwinkle,  // periwinkle
        Color.graphite,    // graphite
    ]

    /// FX overlay accent, matching the web performance graph's FX line.
    static let fx = Color.ochre

    /// Earnings-event accent, matching the web Events card.
    static let earnings = Color.plum

    static let controlRadius: CGFloat = 8    // buttons, chips, rows, inputs
    static let insetRadius: CGFloat = 12     // panels inside a card, menus
    static let cardRadius: CGFloat = 16      // every card, every modal
    static let heroRadius: CGFloat = 20      // one per screen, at most
    static let gutter: CGFloat = 16

    /// Control heights — three steps: toolbar, form, touch.
    static let controlCompact: CGFloat = 28
    static let controlDefault: CGFloat = 36
    static let controlTouch: CGFloat = 44

    /// Card depth tiers. The hero floats highest; insets sit flush inside a card.
    enum Tier { case hero, standard, inset }
}

/// Shared card chrome — Ledger's three tiers, the twin of `.card-inset`,
/// `.card-standard` and `.card-hero` in globals.css:
///
/// - inset: muted fill, no rule — a panel inside a card
/// - standard: white card, 1pt hairline rule, no shadow
/// - hero: the rule plus a soft neutral lift (light appearance only)
///
/// No shine gradient and no tinted shadow: Ledger lifts nothing with colour.
struct CardStyle: ViewModifier {
    var tier: Theme.Tier = .standard
    @Environment(\.colorScheme) private var colorScheme

    private var radius: CGFloat {
        switch tier {
        case .hero: return Theme.heroRadius
        case .standard: return Theme.cardRadius
        case .inset: return Theme.insetRadius
        }
    }

    func body(content: Content) -> some View {
        let shape = RoundedRectangle(cornerRadius: radius, style: .continuous)
        content
            // Clip content (e.g. full-bleed chart fills) to the card's rounded corners.
            .clipShape(shape)
            .overlay(
                shape
                    .strokeBorder(tier == .inset ? Color.clear : Color.cardBorder, lineWidth: 1)
                    .allowsHitTesting(false)
            )
            // The lift is cast by the card's own shape, behind the content. A
            // `.shadow` on the whole card would shadow every chip, pill and
            // label inside it separately.
            .background(
                shape
                    .fill(tier == .inset ? Color.inset : Color.cardBg)
                    .shadow(
                        color: tier == .hero && colorScheme == .light
                            ? Color(red: 0.086, green: 0.09, blue: 0.106).opacity(0.06) : .clear,
                        radius: 12, x: 0, y: 6
                    )
            )
    }
}

extension View {
    /// Apply the shared card chrome at a given depth tier.
    func card(_ tier: Theme.Tier = .standard) -> some View { modifier(CardStyle(tier: tier)) }
}

/// Overline — the small caps label above a figure or a group ("PORTFOLIO
/// VALUE"). 11pt semibold, light tracking; the twin of `.section-label`.
struct SectionLabel: View {
    let title: String
    /// Headers that are too long for a compact width can opt into wrapping
    /// instead of being truncated mid-word.
    var lineLimit: Int = 1
    var body: some View {
        Text(title)
            .appFont(.system(size: 11, weight: .semibold))
            .tracking(0.66)
            .textCase(.uppercase)
            .foregroundStyle(Color.sectionText)
            .lineLimit(lineLimit)
            .fixedSize(horizontal: false, vertical: true)
    }
}

/// Ledger badge: a 6pt-radius tag on a tinted fill, no rule — the twin of the
/// web `Badge`. Gain and loss badges still carry their sign in the text.
struct SemanticBadge: View {
    let text: String
    var tint: Color = .brandIndigo
    var isPositive: Bool? = nil

    private var effectiveTint: Color {
        if let pos = isPositive {
            return pos ? Color.up : Color.down
        }
        return tint
    }

    var body: some View {
        Text(text)
            .appFont(.system(size: 12, weight: .semibold))
            .monospacedDigit()
            .foregroundStyle(effectiveTint)
            .padding(.horizontal, 7)
            .padding(.vertical, 2)
            .background(effectiveTint.opacity(0.12), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
    }
}

extension Color {
    init(hex: UInt) {
        self.init(.sRGB, red: Double((hex >> 16) & 0xff) / 255, green: Double((hex >> 8) & 0xff) / 255,
                  blue: Double(hex & 0xff) / 255, opacity: 1)
    }
}

/// Compact, consistent empty-state shown inside a card body (muted icon + caption).
struct EmptyHint: View {
    let text: String
    var systemImage: String = "tray"
    var body: some View {
        VStack(spacing: 6) {
            Image(systemName: systemImage).appFont(.title3).foregroundStyle(.tertiary)
            Text(text).appFont(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 14)
    }
}

/// Subtle hover highlight for clickable list rows.
private struct RowHover: ViewModifier {
    @State private var hovering = false
    func body(content: Content) -> some View {
        content
            .background(hovering ? Color.inset : .clear,
                        in: RoundedRectangle(cornerRadius: 8))
            .onHover { hovering = $0 }
    }
}

extension View {
    func rowHover() -> some View { modifier(RowHover()) }
}

/// A unified modifier that applies the iOS 26 / macOS 16 Liquid Glass effect
/// if available, and falls back to a standard material or bar background otherwise.
struct LiquidGlassModifier: ViewModifier {
    var interactive: Bool = false

    func body(content: Content) -> some View {
        if #available(iOS 26.0, macOS 16.0, *) {
            if interactive {
                content.glassEffect(.regular.interactive())
            } else {
                content.glassEffect()
            }
        } else {
            if interactive {
                content
            } else {
                content.background(.bar)
            }
        }
    }
}

extension View {
    /// Applies the Liquid Glass container effect if supported by the OS,
    /// otherwise falls back to a standard bar background.
    func liquidGlass() -> some View {
        modifier(LiquidGlassModifier())
    }

    /// Applies the interactive Liquid Glass effect if supported by the OS.
    /// Use on buttons, toggles, and menus inside a glass container.
    func interactiveGlass() -> some View {
        modifier(LiquidGlassModifier(interactive: true))
    }

    /// Requests the decimal-pad keyboard for numeric text fields on iOS.
    /// No-op on macOS, where `keyboardType` is unavailable.
    @ViewBuilder func decimalKeyboard() -> some View {
        #if os(iOS)
        self.keyboardType(.decimalPad)
        #else
        self
        #endif
    }
}

