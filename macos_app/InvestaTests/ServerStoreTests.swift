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
}
