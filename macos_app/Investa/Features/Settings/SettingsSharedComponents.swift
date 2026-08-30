import SwiftUI

// MARK: - Category glyph

/// One monochrome line glyph for a Settings category or row.
///
/// This replaced a filled, gradient-tinted tile in one of five colours
/// (indigo / blue / green / purple / cyan). A category is not a colour —
/// `Theme.dataPalette` is reserved for data, and chrome carries one accent, so
/// the glyph is secondary until its row is active, then it is `Color.brand`.
struct SettingsIcon: View {
    let icon: String
    var size: CGFloat = 22
    var isActive: Bool = false

    var body: some View {
        Image(systemName: icon)
            .appFont(.system(size: size * 0.78, weight: .medium))
            .foregroundStyle(isActive ? Color.brand : Color.secondary)
            .frame(width: size, height: size)
    }
}

// MARK: - Count badge

/// Small indigo count pill for a card head or a row — mirrors the web
/// `countBadgeClassName`.
struct SettingsCountBadge: View {
    let value: Int

    var body: some View {
        Text("\(value)")
            .appFont(.system(size: 11, weight: .bold))
            .monospacedDigit()
            .foregroundStyle(Color.brandInk)
            .padding(.horizontal, 8)
            .padding(.vertical, 2)
            .background(Color.brand.opacity(0.14), in: Capsule())
    }
}

// MARK: - Grouped rows (iPhone hub)

/// A card holding a stack of rows separated by hairlines — the phone's
/// equivalent of an inset-grouped `List` section, drawn in the app's own card
/// chrome instead of the system's grey grouped background.
struct SettingsRowGroup<Content: View>: View {
    @ViewBuilder var content: Content

    var body: some View {
        VStack(spacing: 0) {
            content
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }
}

/// Hairline between two rows in a `SettingsRowGroup`, inset past the glyph.
struct SettingsRowDivider: View {
    var body: some View {
        Rectangle()
            .fill(Color.cardBorder.opacity(0.6))
            .frame(height: 1)
            .padding(.leading, 48)
    }
}

/// A row inside a `SettingsRowGroup` that pushes a detail screen.
struct SettingsNavRow<Destination: View>: View {
    let icon: String
    let title: String
    var subtitle: String? = nil
    var count: Int? = nil
    @ViewBuilder let destination: Destination

    var body: some View {
        NavigationLink {
            destination
        } label: {
            HStack(spacing: 12) {
                SettingsIcon(icon: icon)

                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .appFont(.body)
                        .foregroundStyle(.primary)

                    if let subtitle, !subtitle.isEmpty {
                        Text(subtitle)
                            .appFont(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Spacer(minLength: 8)

                if let count, count > 0 { SettingsCountBadge(value: count) }

                Image(systemName: "chevron.right")
                    .appFont(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.tertiary)
            }
            // One line, in full, at every Dynamic Type size — applied to the
            // stack so a value added later cannot opt out.
            .lineLimit(1)
            .minimumScaleFactor(0.75)
            .padding(.horizontal, 14)
            .frame(minHeight: Theme.controlTouch, alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

/// A row inside a `SettingsRowGroup` that performs an action instead of
/// pushing — sign out, and anything else terminal.
struct SettingsActionRow: View {
    let icon: String
    let title: String
    var tint: Color = .primary
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Image(systemName: icon)
                    .appFont(.system(size: 17, weight: .medium))
                    .foregroundStyle(tint)
                    .frame(width: 22, height: 22)
                Text(title)
                    .appFont(.body)
                    .foregroundStyle(tint)
                Spacer(minLength: 0)
            }
            .lineLimit(1)
            .minimumScaleFactor(0.75)
            .padding(.horizontal, 14)
            .frame(minHeight: Theme.controlTouch, alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Tag Badges & Flow Chips

struct RemovableTagChip: View {
    let text: String
    var color: Color = .brand
    let onRemove: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            Text(text)
                .appFont(.caption.weight(.semibold))
                .foregroundStyle(color)
                .lineLimit(1)

            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill")
                    .appFont(.caption2)
                    .foregroundStyle(color.opacity(0.7))
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(color.opacity(0.12), in: Capsule())
        .overlay(Capsule().strokeBorder(color.opacity(0.25), lineWidth: 1))
    }
}

struct FlowChipsRemovable: View {
    let items: [String]
    var color: Color = .brand
    let onRemove: (String) -> Void

    var body: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 90), spacing: 8)], alignment: .leading, spacing: 8) {
            ForEach(items, id: \.self) { item in
                RemovableTagChip(text: item, color: color) {
                    onRemove(item)
                }
            }
        }
    }
}

// MARK: - Form Helper Modifiers

struct SettingsFieldLabel<Content: View>: View {
    let label: String
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            SectionLabel(title: label)
            content()
        }
    }
}

extension View {
    @ViewBuilder
    func uppercaseAutoCapitalization() -> some View {
        #if os(iOS)
        self.textInputAutocapitalization(.characters)
        #else
        self
        #endif
    }
}
