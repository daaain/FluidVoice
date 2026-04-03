import Foundation

/// Generic transcription provider for Gemma 4 multimodal models with native audio.
///
/// Supports any backend that exposes the Gemini `generateContent` API:
/// - Google Gemini API (cloud)
/// - llama.cpp server (`--mmproj` mode)
/// - MLX server (`mlx_vlm.server`)
///
/// Audio is sent as base64-encoded WAV inline data using a rolling window for
/// streaming preview and full audio for final transcription. When filler word
/// removal is enabled the prompt instructs the model to clean the output in a
/// single inference pass.
final class Gemma4Provider: TranscriptionProvider {

    // MARK: - Types

    enum Gemma4Error: LocalizedError {
        case missingEndpoint
        case invalidAudioData
        case requestFailed(statusCode: Int, body: String)
        case emptyResponse
        case malformedResponse(detail: String)
        case networkError(underlying: Error)

        var errorDescription: String? {
            switch self {
            case .missingEndpoint:
                return "Gemma 4 endpoint is not configured. Set the base URL in Settings > Voice Engine."
            case .invalidAudioData:
                return "Failed to encode audio samples to WAV format."
            case let .requestFailed(statusCode, body):
                return "Gemma 4 API request failed (HTTP \(statusCode)): \(body)"
            case .emptyResponse:
                return "Gemma 4 returned an empty transcription."
            case let .malformedResponse(detail):
                return "Could not parse Gemma 4 response: \(detail)"
            case let .networkError(underlying):
                return "Network error: \(underlying.localizedDescription)"
            }
        }
    }

    // MARK: - Configuration

    /// Maximum seconds of audio to send for streaming preview (rolling window).
    /// Keeps latency bounded while giving the model enough context.
    private let streamingPreviewMaxSeconds: Double = 15

    /// The model identifier sent in the API request.
    private let modelID: String

    // MARK: - TranscriptionProvider

    var name: String { "Gemma 4" }
    var isAvailable: Bool { true }
    var isReady: Bool { self.resolvedEndpoint != nil }

    // MARK: - Init

    init(modelID: String = "gemma-4-e2b") {
        self.modelID = modelID
    }

    // MARK: - Lifecycle

    func prepare(progressHandler: ((Double) -> Void)?) async throws {
        guard self.resolvedEndpoint != nil else {
            throw Gemma4Error.missingEndpoint
        }
        progressHandler?(1.0)
    }

    func modelsExistOnDisk() -> Bool {
        self.resolvedEndpoint != nil
    }

    func clearCache() async throws {}

    // MARK: - Transcription

    func transcribe(_ samples: [Float]) async throws -> ASRTranscriptionResult {
        try await self.performTranscription(samples: samples, removeFillers: self.fillerRemovalEnabled)
    }

    func transcribeStreaming(_ samples: [Float]) async throws -> ASRTranscriptionResult {
        let previewSamples = self.previewWindow(from: samples)
        return try await self.performTranscription(samples: previewSamples, removeFillers: false)
    }

    func transcribeFinal(_ samples: [Float]) async throws -> ASRTranscriptionResult {
        try await self.performTranscription(samples: samples, removeFillers: self.fillerRemovalEnabled)
    }

    // MARK: - Core API Call

    private func performTranscription(samples: [Float], removeFillers: Bool) async throws -> ASRTranscriptionResult {
        guard let endpoint = self.resolvedEndpoint else {
            throw Gemma4Error.missingEndpoint
        }

        let wavData = try Self.encodeWAV(samples: samples, sampleRate: 16_000)
        let base64Audio = wavData.base64EncodedString()

        let prompt: String
        if removeFillers {
            prompt = """
            Transcribe the audio exactly. Remove filler words (um, uh, er, ah, like, \
            you know, I mean, sort of, basically). Output only the clean text.
            """
        } else {
            prompt = "Transcribe the audio exactly. Output only the transcription text."
        }

        let body = Self.buildRequestBody(base64Audio: base64Audio, prompt: prompt)
        let urlString = "\(endpoint)/models/\(self.modelID):generateContent\(self.apiKeyQueryParam)"
        guard let url = URL(string: urlString) else {
            throw Gemma4Error.requestFailed(statusCode: 0, message: "Invalid URL: \(urlString)")
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        request.timeoutInterval = 30

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw Gemma4Error.networkError(underlying: error)
        }

        guard let httpResponse = response as? HTTPURLResponse else {
            throw Gemma4Error.requestFailed(statusCode: 0, body: "Non-HTTP response")
        }
        guard httpResponse.statusCode == 200 else {
            let responseBody = String(data: data, encoding: .utf8) ?? "<unreadable>"
            throw Gemma4Error.requestFailed(statusCode: httpResponse.statusCode, body: responseBody)
        }

        let text = try Self.parseResponse(data)
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw Gemma4Error.emptyResponse }
        return ASRTranscriptionResult(text: trimmed)
    }

    // MARK: - Rolling Window

    /// Extracts a rolling preview window from the end of the audio buffer.
    /// This keeps streaming latency bounded while giving the model enough context
    /// to produce coherent partial transcriptions.
    private func previewWindow(from samples: [Float]) -> [Float] {
        let maxSamples = Int(self.streamingPreviewMaxSeconds * 16_000)
        guard samples.count > maxSamples else { return samples }
        return Array(samples.suffix(maxSamples))
    }

    // MARK: - Endpoint Resolution

    /// Resolves the API endpoint from settings. Supports:
    /// - Google Gemini API (uses the existing "google" provider key and base URL)
    /// - Custom local endpoint (llama.cpp, MLX) via a future gemma4-specific setting
    private var resolvedEndpoint: String? {
        // Use the Google provider's base URL, stripping the "/openai" suffix
        // that ModelRepository adds for the OpenAI-compatible chat endpoint.
        let googleBase = ModelRepository.shared.defaultBaseURL(for: "google")
        let base = googleBase.replacingOccurrences(of: "/openai", with: "")

        // Check if API key is available (cloud) or endpoint is local
        if let key = SettingsStore.shared.getAPIKey(for: "google"), !key.isEmpty {
            return base
        }

        // TODO: Support custom local endpoint setting for llama.cpp / MLX servers
        // e.g. SettingsStore.shared.gemma4Endpoint ?? "http://localhost:8080/v1beta"
        return nil
    }

    /// Returns `?key=<apiKey>` for cloud Gemini, or empty string for local servers.
    private var apiKeyQueryParam: String {
        guard let key = SettingsStore.shared.getAPIKey(for: "google"), !key.isEmpty else {
            return ""
        }
        return "?key=\(key)"
    }

    /// Whether filler word removal is enabled in settings.
    private var fillerRemovalEnabled: Bool {
        SettingsStore.shared.removeFillerWordsEnabled
    }

    // MARK: - Request Building

    static func buildRequestBody(base64Audio: String, prompt: String) -> [String: Any] {
        [
            "contents": [
                [
                    "parts": [
                        [
                            "inline_data": [
                                "mime_type": "audio/wav",
                                "data": base64Audio,
                            ],
                        ],
                        ["text": prompt],
                    ],
                ],
            ],
            "generationConfig": [
                "temperature": 0.0,
                "maxOutputTokens": 4096,
            ],
        ]
    }

    // MARK: - Response Parsing

    /// Parses a Gemini `generateContent` response.
    /// Shape: `{ "candidates": [{ "content": { "parts": [{ "text": "..." }] } }] }`
    static func parseResponse(_ data: Data) throws -> String {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw Gemma4Error.malformedResponse(detail: "Top-level JSON is not a dictionary")
        }
        guard let candidates = json["candidates"] as? [[String: Any]], let first = candidates.first else {
            throw Gemma4Error.malformedResponse(detail: "Missing or empty 'candidates'")
        }
        guard let content = first["content"] as? [String: Any] else {
            throw Gemma4Error.malformedResponse(detail: "Missing 'content' in candidate")
        }
        guard let parts = content["parts"] as? [[String: Any]], !parts.isEmpty else {
            throw Gemma4Error.malformedResponse(detail: "Missing or empty 'parts'")
        }
        let texts = parts.compactMap { $0["text"] as? String }
        guard !texts.isEmpty else {
            throw Gemma4Error.malformedResponse(detail: "No text parts found")
        }
        return texts.joined(separator: " ")
    }

    // MARK: - WAV Encoding

    /// Encodes 16 kHz mono Float32 samples as a 16-bit PCM WAV file.
    static func encodeWAV(samples: [Float], sampleRate: Int) throws -> Data {
        let numChannels: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let byteRate = UInt32(sampleRate) * UInt32(numChannels) * UInt32(bitsPerSample / 8)
        let blockAlign = numChannels * (bitsPerSample / 8)
        let dataSize = UInt32(samples.count * 2)
        let chunkSize: UInt32 = 36 + dataSize

        var d = Data(capacity: 44 + Int(dataSize))

        // RIFF header
        d.append(contentsOf: [0x52, 0x49, 0x46, 0x46]) // "RIFF"
        withUnsafeBytes(of: chunkSize.littleEndian) { d.append(contentsOf: $0) }
        d.append(contentsOf: [0x57, 0x41, 0x56, 0x45]) // "WAVE"

        // fmt sub-chunk
        d.append(contentsOf: [0x66, 0x6D, 0x74, 0x20]) // "fmt "
        withUnsafeBytes(of: UInt32(16).littleEndian) { d.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt16(1).littleEndian) { d.append(contentsOf: $0) }  // PCM
        withUnsafeBytes(of: numChannels.littleEndian) { d.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { d.append(contentsOf: $0) }
        withUnsafeBytes(of: byteRate.littleEndian) { d.append(contentsOf: $0) }
        withUnsafeBytes(of: blockAlign.littleEndian) { d.append(contentsOf: $0) }
        withUnsafeBytes(of: bitsPerSample.littleEndian) { d.append(contentsOf: $0) }

        // data sub-chunk
        d.append(contentsOf: [0x64, 0x61, 0x74, 0x61]) // "data"
        withUnsafeBytes(of: dataSize.littleEndian) { d.append(contentsOf: $0) }

        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let int16 = Int16(clamped * Float(Int16.max))
            withUnsafeBytes(of: int16.littleEndian) { d.append(contentsOf: $0) }
        }

        return d
    }
}
