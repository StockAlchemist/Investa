import Foundation

@MainActor
final class TransactionsViewModel: ObservableObject {
    @Published var transactions: [Transaction] = []
    @Published var isLoading = false
    @Published var errorMessage: String?

    private let api: APIClient
    private var loadTask: Task<Void, Never>?

    init(api: APIClient = .shared) {
        self.api = api
    }

    func reload(accounts: [String]?) {
        loadTask?.cancel()
        loadTask = Task { [weak self] in await self?.load(accounts: accounts) }
    }

    private func load(accounts: [String]?) async {
        isLoading = true
        errorMessage = nil
        defer { isLoading = false }
        do {
            let items: [Transaction] = try await api.get(
                "/transactions", query: APIClient.arrayQuery("accounts", accounts)
            )
            if Task.isCancelled { return }
            transactions = items
        } catch is CancellationError {
            return
        } catch let error as APIError {
            if case .unauthorized = error { return }
            errorMessage = error.errorDescription
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Add or update. When `tx.id` is set it's an update (PUT), else create (POST).
    /// Returns true on success.
    func save(_ tx: Transaction) async -> Bool {
        do {
            if let id = tx.id {
                let _: StatusResponse = try await api.send(
                    method: "PUT", path: "/transactions/\(id)", body: tx
                )
            } else {
                let _: StatusResponse = try await api.send(
                    method: "POST", path: "/transactions", body: tx
                )
            }
            return true
        } catch let error as APIError {
            errorMessage = error.errorDescription
            return false
        } catch {
            errorMessage = error.localizedDescription
            return false
        }
    }

    func delete(_ tx: Transaction) async {
        guard let id = tx.id else { return }
        do {
            let _: StatusResponse = try await api.send(
                method: "DELETE", path: "/transactions/\(id)"
            )
            transactions.removeAll { $0.id == id }
        } catch let error as APIError {
            errorMessage = error.errorDescription
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

/// Generic backend `{ "status": ..., "message"?: ..., "id"?: ... }` response.
struct StatusResponse: Codable, Sendable {
    let status: String?
    let message: String?
    let id: Int?
}
