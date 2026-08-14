import XCTest
@testable import Investa

@MainActor
final class ToastManagerTests: XCTestCase {

    override func setUp() {
        super.setUp()
        ToastManager.shared.dismiss()
    }

    override func tearDown() {
        ToastManager.shared.dismiss()
        super.tearDown()
    }

    func testShowAndDismissToast() {
        let manager = ToastManager.shared
        XCTAssertNil(manager.currentToast)

        manager.show(message: "Test error notification", style: .error, duration: 5.0)

        XCTAssertNotNil(manager.currentToast)
        XCTAssertEqual(manager.currentToast?.message, "Test error notification")
        XCTAssertEqual(manager.currentToast?.style, .error)

        manager.dismiss()
        XCTAssertNil(manager.currentToast)
    }

    func testToastStylesAndIcons() {
        XCTAssertEqual(ToastStyle.error.icon, "exclamationmark.triangle.fill")
        XCTAssertEqual(ToastStyle.warning.icon, "wifi.slash")
        XCTAssertEqual(ToastStyle.info.icon, "info.circle.fill")
        XCTAssertEqual(ToastStyle.success.icon, "checkmark.circle.fill")
    }

    func testNotificationTriggeredToast() {
        let manager = ToastManager.shared
        XCTAssertNil(manager.currentToast)

        NotificationCenter.default.post(
            name: .showToast,
            object: nil,
            userInfo: ["message": "Global toast from notification", "style": ToastStyle.warning]
        )

        let exp = expectation(description: "Toast presented via notification")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            if manager.currentToast?.message == "Global toast from notification" {
                exp.fulfill()
            }
        }

        wait(for: [exp], timeout: 1.0)
        XCTAssertEqual(manager.currentToast?.style, .warning)
    }
}
