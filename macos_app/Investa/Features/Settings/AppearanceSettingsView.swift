import SwiftUI

/// Light / Dark / System, stored locally.
///
/// The pair of `@AppStorage` keys is the app-wide appearance state read by
/// `MainView.preferredColorScheme`: `appearanceSet == false` means "follow the
/// system", and only once the user picks a side does `forceDark` decide it.
enum AppearanceChoice: String, CaseIterable, Identifiable {
    case system, light, dark

    var id: String { rawValue }

    var label: String {
        switch self {
        case .system: return "System"
        case .light: return "Light"
        case .dark: return "Dark"
        }
    }

    var icon: String {
        switch self {
        case .system: return "circle.lefthalf.filled"
        case .light: return "sun.max"
        case .dark: return "moon"
        }
    }

    var hint: String {
        switch self {
        case .system: return "Follow the device appearance"
        case .light: return "Always use the light theme"
        case .dark: return "Always use the dark theme"
        }
    }
}

/// Appearance lives in Settings, not in the chrome: the theme is a preference
/// the user sets once, so it belongs beside the other preferences rather than
/// occupying a permanent slot in the sidebar footer and the control bar's
/// overflow menu. Mirrors `AppearanceTab.tsx` in the web app.
struct AppearanceSettingsView: View {
    var embedded: Bool = false

    @AppStorage("investa.forceDark") private var forceDark = false
    @AppStorage("investa.appearanceSet") private var appearanceSet = false
    @Environment(\.colorScheme) private var colorScheme

    private var choice: AppearanceChoice {
        guard appearanceSet else { return .system }
        return forceDark ? .dark : .light
    }

    var body: some View {
        Group {
            if embedded {
                mainContent
            } else {
                ScrollView {
                    mainContent
                        .padding(16)
                }
                .navigationTitle("Appearance")
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
            }
        }
    }

    private var mainContent: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                SectionLabel(title: "Theme")
                Spacer(minLength: 0)
            }

            Text("Choose the light or dark theme, or follow whatever the device is set to.")
                .appFont(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            // A row of chips rather than a segmented picker: it wraps to the
            // width it is given instead of demanding one.
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) { chips }
                VStack(alignment: .leading, spacing: 8) { chips }
            }

            if choice == .system {
                Text("Currently showing the \(colorScheme == .dark ? "dark" : "light") theme.")
                    .appFont(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(18)
        .card()
    }

    @ViewBuilder private var chips: some View {
        ForEach(AppearanceChoice.allCases) { option in
            chip(option)
        }
    }

    private func chip(_ option: AppearanceChoice) -> some View {
        let isActive = choice == option
        return Button {
            select(option)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: option.icon)
                    .font(.system(size: 12, weight: .semibold))
                Text(option.label)
                    .appFont(.system(size: 12, weight: .semibold))
            }
            .lineLimit(1)
            .minimumScaleFactor(0.85)
            .foregroundStyle(isActive ? Color.brandInk : Color.secondary)
            .padding(.horizontal, 12)
            .frame(height: 32)
            .background(
                RoundedRectangle(cornerRadius: Theme.controlRadius, style: .continuous)
                    .fill(isActive ? Color.brand.opacity(0.12) : Color.clear)
            )
            .overlay(
                RoundedRectangle(cornerRadius: Theme.controlRadius, style: .continuous)
                    .strokeBorder(isActive ? Color.brand.opacity(0.25) : Color.cardBorder, lineWidth: 1)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(option.hint)
        .accessibilityAddTraits(isActive ? [.isSelected] : [])
    }

    private func select(_ option: AppearanceChoice) {
        switch option {
        case .system: appearanceSet = false
        case .light: appearanceSet = true; forceDark = false
        case .dark: appearanceSet = true; forceDark = true
        }
    }
}
