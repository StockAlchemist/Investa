import Foundation

/// A type-erased JSON value used for backend payloads that carry dynamic keys
/// (e.g. currency-suffixed columns like `"Market Value (USD)"`) or otherwise
/// untyped fields that don't map cleanly onto fixed `Codable` structs.
enum JSONValue: Codable, Sendable, Equatable {
    case string(String)
    case double(Double)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let b = try? container.decode(Bool.self) {
            self = .bool(b)
        } else if let d = try? container.decode(Double.self) {
            self = .double(d)
        } else if let s = try? container.decode(String.self) {
            self = .string(s)
        } else if let o = try? container.decode([String: JSONValue].self) {
            self = .object(o)
        } else if let a = try? container.decode([JSONValue].self) {
            self = .array(a)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container, debugDescription: "Unsupported JSON value")
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let s): try container.encode(s)
        case .double(let d): try container.encode(d)
        case .bool(let b): try container.encode(b)
        case .object(let o): try container.encode(o)
        case .array(let a): try container.encode(a)
        case .null: try container.encodeNil()
        }
    }

    // MARK: - Convenience accessors

    var doubleValue: Double? {
        switch self {
        case .double(let d): return d
        case .bool(let b): return b ? 1 : 0
        case .string(let s): return Double(s)
        default: return nil
        }
    }

    var stringValue: String? {
        switch self {
        case .string(let s): return s
        case .double(let d): return String(d)
        case .bool(let b): return String(b)
        default: return nil
        }
    }

    var boolValue: Bool? {
        switch self {
        case .bool(let b): return b
        case .double(let d): return d != 0
        default: return nil
        }
    }

    var arrayValue: [JSONValue]? {
        if case .array(let a) = self { return a }
        return nil
    }

    var objectValue: [String: JSONValue]? {
        if case .object(let o) = self { return o }
        return nil
    }

    subscript(_ key: String) -> JSONValue? {
        if case .object(let o) = self { return o[key] }
        return nil
    }
}
