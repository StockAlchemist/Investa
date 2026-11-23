import XCTest
@testable import Investa

final class ServerStoreTests: XCTestCase {
    /// Saved servers are matched against the active URL in this form, so a
    /// stray slash or space must not make the same backend look like two.
    func testNormalizedURLsCompareEqual() {
        XCTAssertEqual(APIConfig.normalized(" http://100.127.10.38:8000/api/ "), "http://100.127.10.38:8000/api")
        XCTAssertEqual(APIConfig.normalized("http://localhost:8000/api//"), "http://127.0.0.1:8000/api")
        XCTAssertEqual(APIConfig.normalized("http://localhost:8000/api"),
                       APIConfig.normalized("http://127.0.0.1:8000/api/"))
    }

    /// An unnamed server is listed by host and port, not by the whole URL.
    func testDefaultNameIsHostAndPort() {
        XCTAssertEqual(ServerStore.defaultName(for: "http://100.127.10.38:8000/api"), "100.127.10.38:8000")
        XCTAssertEqual(ServerStore.defaultName(for: "https://mac.tail1234.ts.net/api"), "mac.tail1234.ts.net")
    }

    func testSavedServerRoundTrips() throws {
        let server = SavedServer(name: "Home", url: "http://192.168.1.10:8000/api")
        let data = try JSONEncoder().encode([server])
        XCTAssertEqual(try JSONDecoder().decode([SavedServer].self, from: data), [server])
    }

    /// Addresses saved while the backend listened on 8000 move to 8800 once;
    /// HTTPS (Tailscale Serve, no port) and other ports are left alone.
    func testLegacyPortMigration() throws {
        XCTAssertEqual(APIConfig.migratedFromLegacyPort("http://100.66.59.98:8000/api"), "http://100.66.59.98:8800/api")
        XCTAssertEqual(APIConfig.migratedFromLegacyPort("http://localhost:8000/api/"), "http://127.0.0.1:8800/api")
        XCTAssertEqual(APIConfig.migratedFromLegacyPort("https://muon.tail1234.ts.net/api"), "https://muon.tail1234.ts.net/api")
        XCTAssertEqual(APIConfig.migratedFromLegacyPort("http://10.0.0.5:9000/api"), "http://10.0.0.5:9000/api")
        XCTAssertEqual(APIConfig.migratedFromLegacyPort("http://10.0.0.5:8000/other"), "http://10.0.0.5:8000/other")

        let suite = "ServerStoreTests.migration.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        defaults.set("http://100.66.59.98:8000/api", forKey: "investa.api.baseURL")
        let saved = [
            SavedServer(name: "100.66.59.98:8000", url: "http://100.66.59.98:8000/api"),
            SavedServer(name: "Home", url: "http://10.108.157.97:8000/api"),
            SavedServer(name: "HTTPS", url: "https://muon.tail1234.ts.net/api"),
        ]
        defaults.set(try JSONEncoder().encode(saved), forKey: "investa.api.savedServers")

        APIConfig.migrateLegacyPortIfNeeded(defaults)

        XCTAssertEqual(defaults.string(forKey: "investa.api.baseURL"), "http://100.66.59.98:8800/api")
        let migrated = try JSONDecoder().decode([SavedServer].self,
                                                from: XCTUnwrap(defaults.data(forKey: "investa.api.savedServers")))
        XCTAssertEqual(migrated.map(\.url), ["http://100.66.59.98:8800/api", "http://10.108.157.97:8800/api",
                                             "https://muon.tail1234.ts.net/api"])
        XCTAssertEqual(migrated.map(\.name), ["100.66.59.98:8800", "Home", "HTTPS"])

        // Once only: a deliberate 8000 address saved afterwards survives.
        defaults.set("http://192.168.1.2:8000/api", forKey: "investa.api.baseURL")
        APIConfig.migrateLegacyPortIfNeeded(defaults)
        XCTAssertEqual(defaults.string(forKey: "investa.api.baseURL"), "http://192.168.1.2:8000/api")
    }
}
