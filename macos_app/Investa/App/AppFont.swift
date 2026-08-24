import SwiftUI
#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

// MARK: - Why the app has its own font type

/// Investa's type is caption-heavy: on macOS `.caption`, `.caption2` and
/// `.footnote` all resolve to **10pt**, and `.body` to 13pt. Those are the sizes
/// AppKit picks for a 13" laptop, and they are what a 27" display shows too —
/// macOS has no Dynamic Type, so nothing grows with the window.
///
/// `RootView` used to ask for the bump with `.dynamicTypeSize(.xLarge)`. That
/// modifier compiles on macOS but does nothing there: rendering `Text` at
/// `.large`, `.xxxLarge` and `.accessibility3` produces byte-identical glyph
/// metrics. The intended app-wide bump never happened.
///
/// So the scale is applied by us, at the point a font is resolved. `AppFont` is
/// a description of a font — a text style *or* a fixed point size, plus the
/// weight/design/italic trim — that only becomes a real `Font` once it can read
/// `\.appFontScale` out of the environment. `View.appFont(_:)` is the only way
/// to apply one, which is why every call site in the app says `.appFont(…)`
/// rather than `.font(…)`.
///
/// At scale 1.0 an `AppFont` resolves to the *same* `Font` the call site used to
/// name, so small windows, iPad and iPhone render exactly as before — including
/// text-style leading and iOS Dynamic Type, which a fixed `.system(size:)` would
/// throw away.

// MARK: - AppFont

/// A font description resolved against the window's font scale at render time.
///
/// Mirrors the `Font` API used across the app (`.caption`, `.headline`,
/// `.system(size:weight:)`, `.weight(_:)`, `.bold()`, `.monospacedDigit()`, …)
/// so call sites read the same as they did before the scale existed.
struct AppFont: Equatable {
    fileprivate enum Base: Equatable {
        /// A semantic text style — scales from the platform's point size for it.
        case style(Font.TextStyle)
        /// A fixed point size as written at the call site.
        case fixed(CGFloat)
    }

    fileprivate var base: Base
    fileprivate var weight: Font.Weight?
    fileprivate var design: Font.Design?
    fileprivate var isItalic = false
    fileprivate var isMonospacedDigit = false
}

// MARK: Semantic styles

extension AppFont {
    static let largeTitle = AppFont(base: .style(.largeTitle))
    static let title = AppFont(base: .style(.title))
    static let title2 = AppFont(base: .style(.title2))
    static let title3 = AppFont(base: .style(.title3))
    static let headline = AppFont(base: .style(.headline))
    static let subheadline = AppFont(base: .style(.subheadline))
    static let body = AppFont(base: .style(.body))
    static let callout = AppFont(base: .style(.callout))
    static let footnote = AppFont(base: .style(.footnote))
    static let caption = AppFont(base: .style(.caption))
    static let caption2 = AppFont(base: .style(.caption2))

    /// Fixed-size system font, e.g. `.system(size: 11, weight: .bold)`.
    static func system(size: CGFloat, weight: Font.Weight? = nil,
                       design: Font.Design? = nil) -> AppFont {
        AppFont(base: .fixed(size), weight: weight, design: design)
    }

    /// Text-style system font, e.g. `.system(.body, design: .monospaced)`.
    static func system(_ style: Font.TextStyle, design: Font.Design? = nil,
                       weight: Font.Weight? = nil) -> AppFont {
        AppFont(base: .style(style), weight: weight, design: design)
    }
}

// MARK: Trim (mirrors the chainable `Font` modifiers)

extension AppFont {
    func weight(_ w: Font.Weight) -> AppFont { with { $0.weight = w } }
    func bold() -> AppFont { weight(.bold) }
    func italic() -> AppFont { with { $0.isItalic = true } }
    func monospacedDigit() -> AppFont { with { $0.isMonospacedDigit = true } }
    func monospaced() -> AppFont { with { $0.design = .monospaced } }

    private func with(_ mutate: (inout AppFont) -> Void) -> AppFont {
        var copy = self
        mutate(&copy)
        return copy
    }
}

// MARK: Resolution

extension AppFont {
    /// The concrete `Font` for a given scale.
    ///
    /// `scale == 1` returns the font the call site names, untouched — a text
    /// style stays a text style, so it keeps its extra leading and (on iOS) its
    /// Dynamic Type response. Only a scaled window trades those for an explicit
    /// point size, which is the only way to multiply a size on macOS.
    func resolved(scale: CGFloat) -> Font {
        var font: Font
        switch base {
        case .style(let style):
            if scale == 1 {
                font = .system(style, design: design ?? .default)
            } else {
                font = .system(size: AppFont.pointSize(for: style) * scale,
                               weight: AppFont.baseWeight(for: style),
                               design: design ?? .default)
            }
        case .fixed(let size):
            font = .system(size: size * scale, weight: .regular, design: design ?? .default)
        }
        if let weight { font = font.weight(weight) }
        if isItalic { font = font.italic() }
        if isMonospacedDigit { font = font.monospacedDigit() }
        return font
    }

    /// `.headline` is the one style AppKit/UIKit ship at a heavier weight; every
    /// other style is regular. Explicit `.weight(_:)` at the call site wins.
    private static func baseWeight(for style: Font.TextStyle) -> Font.Weight {
        style == .headline ? .semibold : .regular
    }

    /// The platform's own point size for a text style, so scale 1 reproduces
    /// today's rendering exactly instead of a hand-copied table drifting from it.
    private static func pointSize(for style: Font.TextStyle) -> CGFloat {
        pointSizes[style] ?? measuredPointSize(for: style)
    }

    /// Built once, on first use — asking AppKit for eleven font descriptors on
    /// every label of every row would be the expensive kind of correct.
    private static let pointSizes: [Font.TextStyle: CGFloat] = {
        let styles: [Font.TextStyle] = [.largeTitle, .title, .title2, .title3, .headline,
                                        .subheadline, .body, .callout, .footnote, .caption, .caption2]
        return Dictionary(uniqueKeysWithValues: styles.map { ($0, measuredPointSize(for: $0)) })
    }()

    private static func measuredPointSize(for style: Font.TextStyle) -> CGFloat {
        #if canImport(UIKit)
        return UIFont.preferredFont(forTextStyle: uiStyle(for: style)).pointSize
        #elseif canImport(AppKit)
        return NSFont.preferredFont(forTextStyle: nsStyle(for: style)).pointSize
        #else
        return 13
        #endif
    }

    #if canImport(UIKit)
    private static func uiStyle(for style: Font.TextStyle) -> UIFont.TextStyle {
        switch style {
        case .largeTitle: return .largeTitle
        case .title: return .title1
        case .title2: return .title2
        case .title3: return .title3
        case .headline: return .headline
        case .subheadline: return .subheadline
        case .callout: return .callout
        case .footnote: return .footnote
        case .caption: return .caption1
        case .caption2: return .caption2
        default: return .body
        }
    }
    #elseif canImport(AppKit)
    private static func nsStyle(for style: Font.TextStyle) -> NSFont.TextStyle {
        switch style {
        case .largeTitle: return .largeTitle
        case .title: return .title1
        case .title2: return .title2
        case .title3: return .title3
        case .headline: return .headline
        case .subheadline: return .subheadline
        case .callout: return .callout
        case .footnote: return .footnote
        case .caption: return .caption1
        case .caption2: return .caption2
        default: return .body
        }
    }
    #endif
}

// MARK: - The scale itself

/// How much bigger than AppKit's defaults the app draws its type, given the
/// width of the window it is drawn in.
///
/// Linear between two anchors and clamped at both: a window at the app's
/// minimum width (900pt) keeps today's sizes, and the bump tops out at +30%
/// once the window is wide enough that further growth would just make a
/// dashboard read like a poster. Quantised to 5% steps so dragging a window
/// edge re-lays-out a handful of times rather than on every pixel.
enum AppFontScale {
    static let referenceWidth: CGFloat = 900
    static let fullScaleWidth: CGFloat = 1800
    static let maximum: CGFloat = 1.30

    static func forWindowWidth(_ width: CGFloat) -> CGFloat {
        #if os(macOS)
        guard width > referenceWidth else { return 1 }
        let progress = (width - referenceWidth) / (fullScaleWidth - referenceWidth)
        let raw = 1 + min(progress, 1) * (maximum - 1)
        // 5% steps: enough to be a visible change, coarse enough that a resize
        // drag doesn't relayout continuously.
        return (raw * 20).rounded() / 20
        #else
        // iOS keeps Dynamic Type as its size control: the phone already shrinks
        // one step in `iPhoneTextScale()`, and scaling iPad here would freeze
        // every semantic font at a fixed point size, overriding the reader's
        // own text-size setting.
        _ = width
        return 1
        #endif
    }
}

private struct AppFontScaleKey: EnvironmentKey {
    static let defaultValue: CGFloat = 1
}

extension EnvironmentValues {
    /// Multiplier applied to every `appFont(_:)` in the subtree.
    var appFontScale: CGFloat {
        get { self[AppFontScaleKey.self] }
        set { self[AppFontScaleKey.self] = newValue }
    }
}

// MARK: - Applying a font

private struct AppFontModifier: ViewModifier {
    @Environment(\.appFontScale) private var scale
    let font: AppFont

    func body(content: Content) -> some View {
        content.font(font.resolved(scale: scale))
    }
}

extension View {
    /// Sets the font for this view, scaled to the window it is drawn in.
    ///
    /// The app-wide replacement for `.font(_:)`. Use it everywhere, so a screen
    /// can't end up with one label pinned at AppKit's 10pt while the row around
    /// it grows.
    func appFont(_ font: AppFont) -> some View {
        modifier(AppFontModifier(font: font))
    }
}

/// Reads the window's width and publishes the font scale for everything inside.
///
/// Measures with `readingContainerWidth` — the *offered* width, from a
/// zero-height sibling — so the reading can never latch onto the width its own
/// scaled-up content demanded. See `WidthReader.swift` for why that distinction
/// is load-bearing.
struct WindowFontScale<Content: View>: View {
    @State private var width: CGFloat = 0
    @ViewBuilder var content: Content

    var body: some View {
        content
            .readingContainerWidth { width = $0 }
            .environment(\.appFontScale, AppFontScale.forWindowWidth(width))
    }
}
