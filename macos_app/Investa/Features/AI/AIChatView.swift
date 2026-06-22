import SwiftUI

/// Conversational portfolio assistant — mirrors the web `AIChat` floating widget.
/// A sparkles launcher overlays every screen; tapping it opens a chat panel that
/// posts to `/chat/message` with rolling history and persists the conversation.
@MainActor
final class AIChatViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var input = ""
    @Published var isLoading = false

    static let welcome = "Hello! I'm Investa AI. How can I help you with your portfolio today?"

    private let api: APIClient
    private let storageKey = "investa.chat.history"

    init(api: APIClient = .shared) {
        self.api = api
        load()
    }

    func send() {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isLoading else { return }
        messages.append(ChatMessage(role: .user, text: text))
        input = ""
        isLoading = true
        // Prior turns only (the new message goes in `message`); cap context to 40.
        let history = Array(messages.dropLast().suffix(40))
        Task {
            defer { isLoading = false }
            do {
                struct Req: Encodable { let message: String; let history: [ChatMessage] }
                struct Resp: Decodable { let response: String }
                let r: Resp = try await api.send(
                    method: "POST", path: "/chat/message",
                    body: Req(message: text, history: history))
                let reply = r.response.trimmingCharacters(in: .whitespacesAndNewlines)
                messages.append(ChatMessage(role: .ai,
                    text: reply.isEmpty ? "I encountered an issue generating a response. Please try again." : reply))
            } catch {
                messages.append(ChatMessage(role: .ai,
                    text: "I'm sorry, I'm having trouble connecting right now. Please try again later."))
            }
            save()
        }
    }

    func clear() {
        messages = []
        UserDefaults.standard.removeObject(forKey: storageKey)
    }

    private func load() {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let saved = try? JSONDecoder().decode([ChatMessage].self, from: data) else { return }
        messages = saved
    }

    private func save() {
        let toSave = Array(messages.suffix(50))
        if let data = try? JSONEncoder().encode(toSave) {
            UserDefaults.standard.set(data, forKey: storageKey)
        }
    }
}

/// Floating launcher button + presented chat panel, overlaid on the app shell.
/// The bubble is draggable — press and drag to reposition it anywhere over the
/// app, and the spot is remembered across launches. A plain tap opens the chat.
struct AIChatLauncher: View {
    @StateObject private var vm = AIChatViewModel()
    @State private var showChat = false

    /// Persisted bubble center, in container points. NaN until the user first
    /// drags it, which is the signal to fall back to the bottom-trailing corner.
    @AppStorage("investa.chatLauncher.x") private var savedX = Double.nan
    @AppStorage("investa.chatLauncher.y") private var savedY = Double.nan
    @State private var drag: CGSize = .zero

    #if os(iOS)
    @Environment(\.horizontalSizeClass) private var hSize
    #endif

    private let diameter: CGFloat = 56
    private let margin: CGFloat = 20

    var body: some View {
        GeometryReader { geo in
            let resting = anchor(in: geo.size)
            let live = clamped(CGPoint(x: resting.x + drag.width, y: resting.y + drag.height), in: geo.size)
            bubble
                .position(live)
                .gesture(
                    DragGesture(minimumDistance: 6)
                        .onChanged { drag = $0.translation }
                        .onEnded { value in
                            let end = clamped(CGPoint(x: resting.x + value.translation.width,
                                                      y: resting.y + value.translation.height),
                                              in: geo.size)
                            savedX = end.x
                            savedY = end.y
                            drag = .zero
                        }
                )
        }
        .sheet(isPresented: $showChat) { AIChatView(vm: vm) }
    }

    private var bubble: some View {
        Image(systemName: "sparkles")
            .font(.title2.weight(.semibold))
            .foregroundStyle(.white)
            .frame(width: diameter, height: diameter)
            .background(
                LinearGradient(colors: [Color(hex: 0x4f46e5), Color(hex: 0x9333ea)],
                               startPoint: .topTrailing, endPoint: .bottomLeading),
                in: Circle())
            .shadow(color: Color(hex: 0x6366f1).opacity(0.4), radius: 12, y: 4)
            .contentShape(Circle())
            .onTapGesture { showChat = true }
            .accessibilityLabel("Open Investa AI Chat")
            .accessibilityHint("Drag to reposition")
    }

    /// The bubble's resting center: the saved spot once set, else the default
    /// bottom-trailing corner.
    private func anchor(in size: CGSize) -> CGPoint {
        if savedX.isFinite, savedY.isFinite {
            return clamped(CGPoint(x: savedX, y: savedY), in: size)
        }
        let r = diameter / 2
        return CGPoint(x: size.width - margin - r, y: size.height - bottomInset - r)
    }

    /// Extra bottom clearance so the default corner clears the iPhone tab bar.
    private var bottomInset: CGFloat {
        #if os(iOS)
        return hSize == .compact ? 70 : margin
        #else
        return margin
        #endif
    }

    /// Keep the whole bubble on-screen with an 8pt edge gutter.
    private func clamped(_ p: CGPoint, in size: CGSize) -> CGPoint {
        let r = diameter / 2 + 8
        return CGPoint(x: min(max(p.x, r), max(r, size.width - r)),
                       y: min(max(p.y, r), max(r, size.height - r)))
    }
}

struct AIChatView: View {
    @ObservedObject var vm: AIChatViewModel
    @Environment(\.dismiss) private var dismiss
    @FocusState private var inputFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            messageList
            Divider()
            inputBar
        }
        #if os(macOS)
        .frame(width: 460, height: 620)
        #endif
    }

    private var header: some View {
        HStack(spacing: 10) {
            Image(systemName: "sparkles")
                .font(.callout.weight(.semibold)).foregroundStyle(.white)
                .frame(width: 32, height: 32)
                .background(LinearGradient(colors: [Color(hex: 0x6366f1), Color(hex: 0xa855f7)],
                                           startPoint: .topLeading, endPoint: .bottomTrailing), in: RoundedRectangle(cornerRadius: 9))
            VStack(alignment: .leading, spacing: 1) {
                Text("Investa AI").font(.headline)
                HStack(spacing: 5) {
                    Circle().fill(.green).frame(width: 6, height: 6)
                    Text(vm.messages.isEmpty ? "ONLINE" : "MEMORY ACTIVE")
                        .font(.system(size: 10, weight: .bold)).tracking(0.8)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
            Button { vm.clear() } label: { Image(systemName: "trash") }
                .buttonStyle(.borderless).foregroundStyle(.secondary)
                .help("Clear conversation")
                .disabled(vm.messages.isEmpty)
            Button { dismiss() } label: { Image(systemName: "xmark") }
                .buttonStyle(.borderless).foregroundStyle(.secondary)
        }
        .padding(16)
    }

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    if vm.messages.isEmpty {
                        bubble(.ai, AIChatViewModel.welcome)
                    }
                    ForEach(vm.messages) { msg in
                        bubble(msg.role, msg.text).id(msg.id)
                    }
                    if vm.isLoading {
                        typingIndicator.id("typing")
                    }
                }
                .padding(16)
            }
            .onChange(of: vm.messages.count) { _, _ in scrollToBottom(proxy) }
            .onChange(of: vm.isLoading) { _, _ in scrollToBottom(proxy) }
        }
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy) {
        withAnimation(.easeOut(duration: 0.2)) {
            if vm.isLoading { proxy.scrollTo("typing", anchor: .bottom) }
            else if let last = vm.messages.last { proxy.scrollTo(last.id, anchor: .bottom) }
        }
    }

    @ViewBuilder
    private func bubble(_ role: ChatMessage.Role, _ text: String) -> some View {
        let isUser = role == .user
        HStack {
            if isUser { Spacer(minLength: 40) }
            VStack(alignment: isUser ? .trailing : .leading, spacing: 3) {
                Text(Self.markdown(text))
                    .font(.callout)
                    .textSelection(.enabled)
                    .foregroundStyle(isUser ? .white : .primary)
                    .padding(.horizontal, 12).padding(.vertical, 9)
                    .background(
                        isUser ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(.background.secondary),
                        in: RoundedRectangle(cornerRadius: 14))
                Text(isUser ? "You" : "Investa AI")
                    .font(.system(size: 10, weight: .medium)).foregroundStyle(.secondary)
            }
            if !isUser { Spacer(minLength: 40) }
        }
    }

    private var typingIndicator: some View {
        HStack {
            HStack(spacing: 5) {
                ProgressView().controlSize(.small)
                Text("Thinking…").font(.caption).foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12).padding(.vertical, 9)
            .background(.background.secondary, in: RoundedRectangle(cornerRadius: 14))
            Spacer(minLength: 40)
        }
    }

    private var inputBar: some View {
        HStack(spacing: 8) {
            TextField("Ask about your portfolio…", text: $vm.input, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...4)
                .focused($inputFocused)
                .onSubmit(vm.send)
                .padding(.horizontal, 12).padding(.vertical, 9)
                .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
            Button(action: vm.send) {
                Image(systemName: "arrow.up")
                    .font(.callout.weight(.bold)).foregroundStyle(.white)
                    .frame(width: 36, height: 36)
                    .background(Color.accentColor, in: Circle())
            }
            .buttonStyle(.plain)
            .disabled(vm.isLoading || vm.input.trimmingCharacters(in: .whitespaces).isEmpty)
        }
        .padding(16)
    }

    /// Inline-markdown render (matches the AI review / stock detail convention).
    static func markdown(_ s: String) -> AttributedString {
        (try? AttributedString(markdown: s, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))) ?? AttributedString(s)
    }
}
