import Foundation

@MainActor
final class TransactionsViewModel: ObservableObject {
    @Published var transactions: [Transaction] = []
    @Published var pendingIbkr: [Transaction] = []
    @Published var isLoading = false
    @Published var isImporting = false
    @Published var errorMessage: String?
    @Published var statusMessage: String?

    private let api: APIClient
    private var loadTask: Task<Void, Never>?

    init(api: APIClient = .shared) {
        self.api = api
    }

    func reload(accounts: [String]?) {
        loadTask?.cancel()
        loadTask = Task { [weak self] in
            await self?.load(accounts: accounts)
            await self?.loadPending()
        }
    }

    // MARK: - Document import + batch

    struct ParseResult: Decodable, Sendable {
        let transactions: [Transaction]?
        let count: Int?
        let message: String?
    }

    /// Parse a PDF/image document into draft transactions for review.
    func parseDocument(_ fileURL: URL) async -> [Transaction] {
        isImporting = true; errorMessage = nil
        do {
            let result: ParseResult = try await api.postMultipart("/transactions/parse_document", fileURL: fileURL)
            let txs = result.transactions ?? []
            isImporting = false
            if txs.isEmpty { statusMessage = result.message ?? "No transactions found in document." }
            return txs
        } catch let error as APIError {
            isImporting = false
            errorMessage = error.errorDescription; return []
        } catch { 
            isImporting = false
            errorMessage = error.localizedDescription; return [] 
        }
    }

    /// Commit reviewed transactions in one batch.
    func addBatch(_ txns: [Transaction], autoAddCash: Bool) async -> Bool {
        struct Body: Encodable { let transactions: [Transaction]; let auto_add_cash: Bool }
        do {
            let res: StatusResponse = try await api.send(method: "POST", path: "/transactions/batch",
                body: Body(transactions: txns, auto_add_cash: autoAddCash))
            statusMessage = "Imported \(res.id ?? txns.count) transactions."
            return true
        } catch let error as APIError { errorMessage = error.errorDescription; return false }
        catch { errorMessage = error.localizedDescription; return false }
    }

    // MARK: - IBKR

    func syncIbkr() async {
        statusMessage = "Syncing IBKR…"
        let _: StatusResponse? = try? await api.send(method: "POST", path: "/sync/ibkr")
        await loadPending()
        statusMessage = pendingIbkr.isEmpty ? "No new IBKR transactions." : "\(pendingIbkr.count) pending IBKR transactions."
    }
    func loadPending() async {
        pendingIbkr = (try? await api.get("/sync/ibkr/pending")) ?? []
    }
    func approvePending(_ ids: [Int]) async {
        let _: StatusResponse? = try? await api.send(method: "POST", path: "/sync/ibkr/approve", body: ids)
        await loadPending(); reload(accounts: nil)
    }
    func rejectPending(_ ids: [Int]) async {
        let _: StatusResponse? = try? await api.send(method: "POST", path: "/sync/ibkr/reject", body: ids)
        await loadPending()
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
