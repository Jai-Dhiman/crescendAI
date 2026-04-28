import Foundation

enum ChatEvent {
    case start(conversationId: String)
    case delta(String)
    case toolStart(toolName: String)
    case toolResult(artifacts: [ArtifactConfig])
    case done
    case error(String)
}

struct SSEParser {
    private enum EventPayload: Decodable {
        case start(conversationId: String)
        case delta(text: String)
        case toolStart(toolName: String)
        case toolResult(componentsJson: String?)
        case done
        case error(message: String)
        case unknown

        private enum CodingKeys: String, CodingKey {
            case type, text, conversationId, componentsJson, toolName, message
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            let type_ = try c.decode(String.self, forKey: .type)
            switch type_ {
            case "start":
                self = .start(conversationId: (try? c.decode(String.self, forKey: .conversationId)) ?? "")
            case "delta":
                self = .delta(text: (try? c.decode(String.self, forKey: .text)) ?? "")
            case "tool_start":
                self = .toolStart(toolName: (try? c.decode(String.self, forKey: .toolName)) ?? "")
            case "tool_result":
                self = .toolResult(componentsJson: try? c.decode(String.self, forKey: .componentsJson))
            case "done":
                self = .done
            case "error":
                self = .error(message: (try? c.decode(String.self, forKey: .message)) ?? "Unknown error")
            default:
                self = .unknown
            }
        }
    }

    static func parse(line: String) -> ChatEvent? {
        guard line.hasPrefix("data: ") else { return nil }
        let payload = String(line.dropFirst(6))

        if payload == "[DONE]" { return .done }

        guard let data = payload.data(using: .utf8),
              let parsed = try? JSONDecoder().decode(EventPayload.self, from: data) else {
            return nil
        }

        switch parsed {
        case .start(let cid): return .start(conversationId: cid)
        case .delta(let text): return .delta(text)
        case .toolStart(let name): return .toolStart(toolName: name)
        case .toolResult(let json):
            guard let json,
                  let data = json.data(using: .utf8),
                  let artifacts = try? JSONDecoder().decode([ArtifactConfig].self, from: data) else {
                return .toolResult(artifacts: [])
            }
            return .toolResult(artifacts: artifacts)
        case .done: return .done
        case .error(let msg): return .error(msg)
        case .unknown: return nil
        }
    }
}

@MainActor
protocol ChatServiceProtocol: AnyObject {
    func send(message: String, conversationId: String?) -> AsyncStream<ChatEvent>
}

@MainActor
@Observable
final class ChatService: ChatServiceProtocol {
    private let session: URLSession

    init(session: URLSession = .shared) {
        self.session = session
    }

    func send(message: String, conversationId: String?) -> AsyncStream<ChatEvent> {
        AsyncStream { continuation in
            Task {
                do {
                    try await streamSSE(message: message, conversationId: conversationId, continuation: continuation)
                } catch {
                    continuation.yield(.error(error.localizedDescription))
                    continuation.finish()
                }
            }
        }
    }

    private func streamSSE(
        message: String,
        conversationId: String?,
        continuation: AsyncStream<ChatEvent>.Continuation
    ) async throws {
        var request = URLRequest(url: APIEndpoints.chat())
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")

        struct ChatBody: Encodable {
            let message: String
            let conversationId: String?
        }
        request.httpBody = try JSONEncoder().encode(ChatBody(message: message, conversationId: conversationId))

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200..<300).contains(httpResponse.statusCode) else {
            throw APIError.invalidResponse((response as? HTTPURLResponse)?.statusCode ?? 0)
        }

        let body = String(decoding: data, as: UTF8.self)
        let blocks = body.components(separatedBy: "\n\n")
        for block in blocks {
            let lines = block.components(separatedBy: "\n")
            for line in lines {
                if let event = SSEParser.parse(line: line) {
                    continuation.yield(event)
                    if case .done = event {
                        continuation.finish()
                        return
                    }
                }
            }
        }
        continuation.finish()
    }
}
