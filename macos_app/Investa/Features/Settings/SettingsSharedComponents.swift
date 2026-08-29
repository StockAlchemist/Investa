import SwiftUI

// MARK: - iOS HIG Settings Row Icon Badge

/// Colored rounded rectangle icon tile matching Apple iOS Settings app design.
struct SettingsIconBadge: View {
    let icon: String
    let color: Color
    var size: CGFloat = 30
    var iconSize: CGFloat = 16

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .fill(color.gradient)
            Image(systemName: icon)
                .font(.system(size: iconSize, weight: .semibold))
                .foregroundStyle(.white)
        }
        .frame(width: size, height: size)
        .shadow(color: color.opacity(0.3), radius: 3, x: 0, y: 1.5)
    }
}

// MARK: - Settings Navigation Link Row (iOS)

/// Reusable navigation row for the iOS Settings Hub with icon badge, title, subtitle, and optional count badge.
struct SettingsNavRow<Destination: View>: View {
    let icon: String
    let iconColor: Color
    let title: String
    var subtitle: String? = nil
    var badge: String? = nil
    @ViewBuilder let destination: Destination

    var body: some View {
        NavigationLink(destination: destination) {
            HStack(spacing: 12) {
                SettingsIconBadge(icon: icon, color: iconColor)

                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .appFont(.body)
                        .foregroundStyle(.primary)

                    if let subtitle = subtitle, !subtitle.isEmpty {
                        Text(subtitle)
                            .appFont(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }

                Spacer()

                if let badge = badge, !badge.isEmpty {
                    Text(badge)
                        .appFont(.caption.weight(.medium))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(Color.primary.opacity(0.06), in: Capsule())
                }
            }
            .padding(.vertical, 2)
        }
    }
}

// MARK: - Reusable Card for Desktop / iPad

struct SettingsCard<Content: View>: View {
    let title: String
    var subtitle: String? = nil
    var icon: String? = nil
    var iconColor: Color? = nil
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .center, spacing: 10) {
                if let icon = icon {
                    SettingsIconBadge(icon: icon, color: iconColor ?? .blue, size: 28, iconSize: 15)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .appFont(.headline.bold())

                    if let subtitle = subtitle {
                        Text(subtitle)
                            .appFont(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }

                Spacer()
            }

            content
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(Color.primary.opacity(0.03))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.02), radius: 8, x: 0, y: 4)
    }
}

// MARK: - Tag Badges & Flow Chips

struct RemovableTagChip: View {
    let text: String
    var color: Color = .blue
    let onRemove: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            Text(text)
                .appFont(.caption.weight(.semibold))
                .foregroundStyle(color)

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
    var color: Color = .blue
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
            Text(label)
                .appFont(.caption2.bold())
                .foregroundStyle(.secondary)
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

