import Foundation
import Combine

/// Owns authentication state: login, logout, session restore, and reacting to
/// 401s broadcast via `.authExpired`.
@MainActor
final class AuthViewModel: ObservableObject {
    enum State: Equatable {
        case checking      // restoring a saved session on launch
        case loggedOut
        case loggedIn(User)
    }

    @Published private(set) var state: State = .checking
    @Published var isSubmitting = false
    @Published var errorMessage: String?

    private let api: APIClient
    private var cancellable: AnyCancellable?

    init(api: APIClient = .shared) {
        self.api = api
        cancellable = NotificationCenter.default
            .publisher(for: .authExpired)
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in self?.handleExpiry() }
    }

    var currentUser: User? {
        if case .loggedIn(let user) = state { return user }
        return nil
    }

    /// On launch: if a token is stored, verify it via `/auth/me`.
    func restoreSession() async {
        guard KeychainStore.loadToken() != nil else {
            state = .loggedOut
            return
        }
        do {
            let user: User = try await api.get("/auth/me")
            state = .loggedIn(user)
        } catch {
            KeychainStore.deleteToken()
            state = .loggedOut
        }
    }

    func login(username: String, password: String) async {
        errorMessage = nil
        isSubmitting = true
        defer { isSubmitting = false }
        do {
            let token: Token = try await api.postForm(
                "/auth/login",
                fields: ["username": username, "password": password]
            )
            KeychainStore.saveToken(token.accessToken)
            let user: User = try await api.get("/auth/me")
            state = .loggedIn(user)
        } catch let error as APIError {
            errorMessage = error.errorDescription
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func logout() {
        KeychainStore.deleteToken()
        state = .loggedOut
    }

    private func handleExpiry() {
        guard case .loggedIn = state else { return }
        KeychainStore.deleteToken()
        errorMessage = "Your session expired. Please log in again."
        state = .loggedOut
    }
}
