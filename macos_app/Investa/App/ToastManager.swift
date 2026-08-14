import SwiftUI
import Combine

/// Visual style of a toast message.
enum ToastStyle: Sendable {
    case error
    case warning
    case info
    case success

    var icon: String {
        switch self {
        case .error: return "exclamationmark.triangle.fill"
        case .warning: return "wifi.slash"
        case .info: return "info.circle.fill"
        case .success: return "checkmark.circle.fill"
        }
    }

    var color: Color {
        switch self {
        case .error: return .red
        case .warning: return .orange
        case .info: return .blue
        case .success: return .green
        }
    }
}

/// Identifiable representation of an active toast notification.
struct ToastItem: Identifiable, Equatable, Sendable {
    let id: UUID
    let message: String
    let style: ToastStyle
    let timestamp: Date
    let duration: TimeInterval

    init(
        id: UUID = UUID(),
        message: String,
        style: ToastStyle = .error,
        timestamp: Date = Date(),
        duration: TimeInterval = 3.5
    ) {
        self.id = id
        self.message = message
        self.style = style
        self.timestamp = timestamp
        self.duration = duration
    }

    static func == (lhs: ToastItem, rhs: ToastItem) -> Bool {
        lhs.id == rhs.id && lhs.message == rhs.message && lhs.duration == rhs.duration
    }
}

extension Notification.Name {
    /// Post this notification to present a toast globally from any thread.
    /// `userInfo` can contain:
    /// - `"message"`: `String`
    /// - `"style"`: `ToastStyle` (optional, defaults to `.error`)
    /// - `"duration"`: `TimeInterval` (optional, defaults to 3.5)
    static let showToast = Notification.Name("investa.showToast")
}

/// Centralized manager for displaying transient toast alerts across the application.
@MainActor
final class ToastManager: ObservableObject {
    static let shared = ToastManager()

    @Published private(set) var currentToast: ToastItem?
    private var dismissTask: Task<Void, Never>?
    private var lastToastTime: [String: Date] = [:]
    private var cancellables = Set<AnyCancellable>()

    init() {
        NotificationCenter.default.publisher(for: .showToast)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] note in
                guard let self = self else { return }
                if let item = note.object as? ToastItem {
                    self.show(item: item)
                } else if let msg = note.userInfo?["message"] as? String {
                    let style = (note.userInfo?["style"] as? ToastStyle) ?? .error
                    let duration = (note.userInfo?["duration"] as? TimeInterval) ?? 3.5
                    self.show(message: msg, style: style, duration: duration)
                }
            }
            .store(in: &cancellables)
    }

    /// Present a toast message with optional deduplication within a short window.
    func show(message: String, style: ToastStyle = .error, duration: TimeInterval = 3.5) {
        let trimmed = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        // Deduplicate identical messages occurring within 2.5 seconds.
        if let last = lastToastTime[trimmed], Date().timeIntervalSince(last) < 2.5 {
            return
        }
        lastToastTime[trimmed] = Date()

        let item = ToastItem(message: trimmed, style: style, duration: duration)
        show(item: item)
    }

    func show(item: ToastItem) {
        dismissTask?.cancel()
        withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) {
            currentToast = item
        }

        dismissTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(item.duration * 1_000_000_000))
            guard !Task.isCancelled else { return }
            await self?.dismiss()
        }
    }

    func dismiss() {
        dismissTask?.cancel()
        dismissTask = nil
        withAnimation(.easeOut(duration: 0.25)) {
            currentToast = nil
        }
    }
}

/// View rendering an active toast notification banner with modern glassmorphism.
struct ToastBannerView: View {
    let item: ToastItem
    let onDismiss: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: item.style.icon)
                .font(.system(size: 16, weight: .bold))
                .foregroundStyle(item.style.color)

            Text(item.message)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.primary)
                .lineLimit(3)
                .multilineTextAlignment(.leading)

            Spacer(minLength: 4)

            Button(action: onDismiss) {
                Image(systemName: "xmark")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(.secondary)
                    .padding(6)
                    .background(Color.primary.opacity(0.06), in: Circle())
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .frame(maxWidth: 480)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                #if os(macOS)
                .fill(Color(nsColor: .windowBackgroundColor).opacity(0.85))
                #else
                .fill(Color(uiColor: .systemBackground).opacity(0.85))
                #endif
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                .shadow(color: Color.black.opacity(0.18), radius: 16, x: 0, y: 6)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(item.style.color.opacity(0.35), lineWidth: 1)
        )
        .padding(.horizontal, 16)
    }
}

/// Modifier that attaches the ToastManager container to any view hierarchy.
struct ToastOverlayModifier: ViewModifier {
    @StateObject private var toastManager = ToastManager.shared

    func body(content: Content) -> some View {
        content
            .overlay(alignment: .top) {
                if let toast = toastManager.currentToast {
                    ToastBannerView(item: toast) {
                        toastManager.dismiss()
                    }
                    .padding(.top, 16)
                    .transition(.move(edge: .top).combined(with: .opacity))
                    .zIndex(9999)
                }
            }
    }
}

extension View {
    /// Adds global toast presentation overlay to this view.
    func toastOverlay() -> some View {
        modifier(ToastOverlayModifier())
    }
}
