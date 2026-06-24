import SwiftUI

/// The app's single dropdown/menu primitive. A tap on `label` opens a
/// content-sized popover (kept as a compact popover on iPhone via
/// `.presentationCompactAdaptation`) instead of the system `Menu`, so every
/// dropdown — currency, accounts, benchmarks, overflow, row actions — shares one
/// look and width behaviour. Build the body from `MenuRow` / `MenuToggleRow` /
/// `MenuSectionHeader` / `MenuDivider` for consistent styling; `MenuRow`
/// auto-dismisses the popover after its action.
struct PopoverMenu<Label: View, Content: View>: View {
    var minWidth: CGFloat = 220
    var maxHeight: CGFloat = 480
    private let content: () -> Content
    private let label: () -> Label

    @State private var open = false

    init(minWidth: CGFloat = 220, maxHeight: CGFloat = 480,
         @ViewBuilder content: @escaping () -> Content,
         @ViewBuilder label: @escaping () -> Label) {
        self.minWidth = minWidth; self.maxHeight = maxHeight
        self.content = content; self.label = label
    }

    var body: some View {
        Button { open.toggle() } label: { label() }
            .buttonStyle(.plain)
            .popover(isPresented: $open) {
                ScrollView {
                    VStack(alignment: .leading, spacing: 1) { content() }
                        .padding(.vertical, 6)
                }
                .scrollBounceBehavior(.basedOnSize)
                .frame(minWidth: minWidth)
                .frame(maxHeight: maxHeight)
                #if os(iOS)
                .presentationCompactAdaptation(.popover)
                .presentationBackground(.regularMaterial)
                #endif
            }
    }
}

/// A tappable menu row. Runs `action` then dismisses the popover.
struct MenuRow: View {
    let title: String
    var systemImage: String? = nil
    var role: ButtonRole? = nil
    var trailing: String? = nil
    let action: () -> Void

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        Button(role: role) { action(); dismiss() } label: {
            HStack(spacing: 10) {
                if let systemImage {
                    Image(systemName: systemImage).frame(width: 20)
                }
                Text(title).fixedSize(horizontal: true, vertical: false)
                Spacer(minLength: 16)
                if let trailing {
                    Text(trailing).foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, 14).padding(.vertical, 9)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(role == .destructive ? Color.red : .primary)
    }
}

/// A menu row with a leading/trailing checkmark for on/off state. Multi-select
/// rows keep the popover open (`dismissOnTap` false); single-select set it true.
struct MenuToggleRow: View {
    let title: String
    var systemImage: String? = nil
    let isOn: Bool
    var dismissOnTap: Bool = false
    var trailing: String? = nil
    let action: () -> Void

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        Button { action(); if dismissOnTap { dismiss() } } label: {
            HStack(spacing: 10) {
                if let systemImage {
                    Image(systemName: systemImage).frame(width: 20)
                }
                Text(title).fixedSize(horizontal: true, vertical: false)
                Spacer(minLength: 16)
                if let trailing {
                    Text(trailing).font(.caption2.weight(.medium)).foregroundStyle(.secondary)
                }
                Image(systemName: "checkmark")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(.tint)
                    .opacity(isOn ? 1 : 0)
            }
            .padding(.horizontal, 14).padding(.vertical, 9)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

/// A small uppercased section header inside a menu.
struct MenuSectionHeader: View {
    let title: String
    init(_ title: String) { self.title = title }
    var body: some View {
        Text(title)
            .font(.caption2.weight(.semibold))
            .foregroundStyle(.secondary)
            .textCase(.uppercase)
            .padding(.horizontal, 14).padding(.top, 8).padding(.bottom, 3)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// A divider sized to the menu's padding.
struct MenuDivider: View {
    var body: some View { Divider().padding(.vertical, 4) }
}
