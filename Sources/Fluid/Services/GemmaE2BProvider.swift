import Foundation

/// Transcription provider that uses Google's Gemini REST API to transcribe audio
/// via the Gemma 4 E2B multimodal model.
///
/// Audio is sent as base64-encoded WAV inline data. The model performs speech-to-text
/// natively, with optional filler word removal handled in the transcription prompt.
final class GemmaE2BProvider: TranscriptionProvider {

    // MARK: - Types

    enum GemmaError: LocalizedError {
        case missingAPIKey
        case invalidAudioData
        case requestFailed(statusCode: Int, message: String)
        case emptyResponse
        case malformedResponse(detail: String)
        case networkError(underlying: Error)

        var errorDescription: String? {
            switch self {
            case .missingAPIKey:
                return "Google API key is not configured. Add your key in Settings > API Keys."
            case .invalidAudioData:
                return "Failed to encode audio samples to WAV format."
            case let .requestFailed(statusCode, message):
                return "Gemini API request failed (HTTP \(statusCode)): \(message)"
            case .emptyResponse:
                return "Gemini API returned an empty transcription."
            case let .malformedResponse(detail):
                return "Could not parse Gemini API response: \(detail)"
            case let .networkError(underlying):
                return "Network error communicating with Gemini API: \(underlying.localizedDescription)"
            }
        }
    }

    // MARK: - Configuration

    static let modelID = "gemma-4-e2b"
    static let apiBase = "https://generativelanguage.googleapis.com/v1beta"

    let removeFillerWordsInline: Bool

    // MARK: - TranscriptionProvider

    var name: String { "Gemma 4 E2B" }

    var isAvailable: Bool { true }

    var isReady: Bool {
        SettingsStore.shared.getAPIKey(for: "google") != nil
    }

    // MARK: - Init

    init(removeFillerWordsInline: Bool = false) {
        self.removeFillerWordsInline = removeFillerWordsInline
    }

    func prepare(progressHandler: ((Double) -> Void)?) async throws {
        guard SettingsStore.shared.getAPIKey(for: "google") != nil else {
            throw GemmaError.missingAPIKey
        }
        progressHandler?(1.0)
    }

    func transcribe(_ samples: [Float]) async throws -> ASRTranscriptionResult {
        guard let apiKey = SettingsStore.shared.getAPIKey(for: "google") else {
            throw GemmaError.missingAPIKey
        }

        let wavData = try GemmaE2BProvider.encodeWAV(samples: samples, sampleRate: 16_000)
        let base64Audio = wavData.base64EncodedString()

        let prompt: String
        if self.removeFillerWordsInline {
            prompt = "Transcribe the following audio exactly. Remove filler words such as um, uh, er, ah, like, you know, and similar hesitations. Output only the clean transcription text with no extra commentary."
        } else {
            prompt = "Transcribe the following audio exactly. Output only the transcription text with no extra commentary."
        }

        let requestBody: [String: Any] = [
            "contents": [
                [
                    "parts": [
                        [
                            "inline_data": [
                                "mime_type": "audio/wav",
                                "data": base64Audio,
                            ],
                        ],
                        [
                            "text": prompt,
                        ],
                    ],
                ],
            ],
        ]

        let urlString = "\(GemmaE2BProvider.apiBase)/models/\(GemmaE2BProvider.modelID):generateContent?key=\(apiKey)"
        guard let url = URL(string: urlString) else {
            throw GemmaError.requestFailed(statusCode: 0, message: "Invalid API URL")
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw GemmaError.networkError(underlying: error)
        }

        guard let httpResponse = response as? HTTPURLResponse else {
            throw GemmaError.requestFailed(statusCode: 0, message: "Non-HTTP response")
        }

        guard httpResponse.statusCode == 200 else {
            let body = String(data: data, encoding: .utf8) ?? "<unreadable>"
            throw GemmaError.requestFailed(statusCode: httpResponse.statusCode, message: body)
        }

        let text = try GemmaE2BProvider.parseGeminiResponse(data)
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw GemmaError.emptyResponse
        }

        return ASRTranscriptionResult(text: trimmed)
    }

    func transcribeStreaming(_ samples: [Float]) async throws -> ASRTranscriptionResult {
        try await self.transcribe(samples)
    }

    func transcribeFinal(_ samples: [Float]) async throws -> ASRTranscriptionResult {
        try await self.transcribe(samples)
    }

    func modelsExistOnDisk() -> Bool {
        return SettingsStore.shared.getAPIKey(for: "google") != nil
    }

    func clearCache() async throws {}

    // MARK: - WAV Encoding

    static func encodeWAV(samples: [Float], sampleRate: Int) throws -> Data {
        let numChannels: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let byteRate = UInt32(sampleRate) * UInt32(numChannels) * UInt32(bitsPerSample / 8)
        let blockAlign = numChannels * (bitsPerSample / 8)
        let dataSize = UInt32(samples.count * Int(bitsPerSample / 8))
        let chunkSize = 36 + dataSize

        var data = Data()
        data.reserveCapacity(44 + Int(dataSize))

        data.append(contentsOf: [0x52, 0x49, 0x46, 0x46])
        data.append(littleEndian: chunkSize)
        data.append(contentsOf: [0x57, 0x41, 0x56, 0x45])

        data.append(contentsOf: [0x66, 0x6D, 0x74, 0x20])
        data.append(littleEndian: UInt32(16))
        data.append(littleEndian: UInt16(1))
        data.append(littleEndian: numChannels)
        data.append(littleEndian: UInt32(sampleRate))
        data.append(littleEndian: byteRate)
        data.append(littleEndian: blockAlign)
        data.append(littleEndian: bitsPerSample)

        data.append(contentsOf: [0x64, 0x61, 0x74, 0x61])
        data.append(littleEndian: dataSize)

        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let int16Value = Int16(clamped * Float(Int16.max))
            data.append(littleEndian: int16Value)
        }

        return data
    }

    // MARK: - Response Parsing

    static func parseGeminiResponse(_ data: Data) throws -> String {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw GemmaError.malformedResponse(detail: "Top-level JSON is not a dictionary")
        }
        guard let candidates = json["candidates"] as? [[String: Any]], !candidates.isEmpty else {
            throw GemmaError.malformedResponse(detail: "Missing or empty 'candidates' array")
        }
        guard let content = candidates[0]["content"] as? [String: Any] else {
            throw GemmaError.malformedResponse(detail: "Missing 'content' in first candidate")
        }
        guard let parts = content["parts"] as? [[String: Any]], !parts.isEmpty else {
            throw GemmaError.malformedResponse(detail: "Missing or empty 'parts' array")
        }
        var texts: [String] = []
        for part in parts {
            if let text = part["text"] as? String {
                texts.append(text)
            }
        }
        guard !texts.isEmpty else {
            throw GemmaError.malformedResponse(detail: "No text parts found in response")
        }
        return texts.joined(separator: " ")
    }
}

private extension Data {
    mutating func append(littleEndian value: UInt16) {
        var le = value.littleEndian
        Swift.withUnsafeBytes(of: &le) { self.append(contentsOf: $0) }
    }
    mutating func append(littleEndian value: UInt32) {
        var le = value.littleEndian
        Swift.withUnsafeBytes(of: &le) { self.append(contentsOf: $0) }
    }
    mutating func append(littleEndian value: Int16) {
        var le = value.littleEndian
        Swift.withUnsafeBytes(of: &le) { self.append(contentsOf: $0) }
    }
}
