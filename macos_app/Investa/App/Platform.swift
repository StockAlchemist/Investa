import SwiftUI
#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

// MARK: - Cross-platform image

#if canImport(UIKit)
typealias PlatformImage = UIImage
#elseif canImport(AppKit)
typealias PlatformImage = NSImage
#endif

extension PlatformImage {
    static func decode(_ data: Data) -> PlatformImage? {
        #if canImport(UIKit)
        return UIImage(data: data)
        #else
        return NSImage(data: data)
        #endif
    }
}

extension Image {
    init(platformImage: PlatformImage) {
        #if canImport(UIKit)
        self.init(uiImage: platformImage)
        #else
        self.init(nsImage: platformImage)
        #endif
    }
}

// MARK: - Adaptive color

extension Color {
    /// Appearance-aware color built from platform dynamic colors (sRGB 0…1).
    static func adaptive(light: (r: Double, g: Double, b: Double),
                         dark: (r: Double, g: Double, b: Double)) -> Color {
        #if canImport(UIKit)
        return Color(UIColor { trait in
            trait.userInterfaceStyle == .dark
                ? UIColor(red: dark.r, green: dark.g, blue: dark.b, alpha: 1)
                : UIColor(red: light.r, green: light.g, blue: light.b, alpha: 1)
        })
        #elseif canImport(AppKit)
        return Color(NSColor(name: nil) { appearance in
            appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
                ? NSColor(srgbRed: dark.r, green: dark.g, blue: dark.b, alpha: 1)
                : NSColor(srgbRed: light.r, green: light.g, blue: light.b, alpha: 1)
        })
        #else
        return Color(.sRGB, red: light.r, green: light.g, blue: light.b)
        #endif
    }
}

// MARK: - View helpers

extension View {
    /// Compact, chrome-less menu styling — `.borderlessButton` on macOS, default elsewhere.
    @ViewBuilder func borderlessMenu() -> some View {
        #if os(macOS)
        self.menuStyle(.borderlessButton)
        #else
        self.menuStyle(.automatic)
        #endif
    }

    /// Minimum window size — applied on macOS only (iOS sizes to the device).
    @ViewBuilder func macMinSize(width: CGFloat, height: CGFloat) -> some View {
        #if os(macOS)
        self.frame(minWidth: width, minHeight: height)
        #else
        self
        #endif
    }

    /// Forces the Gregorian calendar for date pickers so they don't render a
    /// locale-specific calendar (e.g. the Buddhist era under a Thai locale).
    /// Transactions are stored as ISO `yyyy-MM-dd`, so the picker should always
    /// show Gregorian years regardless of the user's regional settings.
    func gregorianCalendar() -> some View {
        environment(\.calendar, Calendar(identifier: .gregorian))
    }
}

// MARK: - File export (save panel on macOS, share sheet on iOS)

@MainActor
func exportText(_ text: String, filename: String) {
    #if os(macOS)
    let panel = NSSavePanel()
    panel.nameFieldStringValue = filename
    if panel.runModal() == .OK, let url = panel.url {
        try? text.write(to: url, atomically: true, encoding: .utf8)
    }
    #elseif canImport(UIKit)
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(filename)
    try? text.write(to: url, atomically: true, encoding: .utf8)
    let av = UIActivityViewController(activityItems: [url], applicationActivities: nil)
    guard let scene = UIApplication.shared.connectedScenes.first(where: { $0.activationState == .foregroundActive }) as? UIWindowScene,
          let root = scene.keyWindow?.rootViewController else { return }
    if let pop = av.popoverPresentationController {
        pop.sourceView = root.view
        pop.sourceRect = CGRect(x: root.view.bounds.midX, y: root.view.bounds.midY, width: 0, height: 0)
    }
    root.present(av, animated: true)
    #endif
}
