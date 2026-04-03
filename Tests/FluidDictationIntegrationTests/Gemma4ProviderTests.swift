import Foundation
import XCTest

@testable import FluidVoice_Debug

@MainActor
final class Gemma4ProviderTests: XCTestCase {

    // MARK: - WAV Encoding

    func testEncodeWAV_validHeader() throws {
        let wav = try Gemma4Provider.encodeWAV(samples: [0.0, 0.5, -0.5], sampleRate: 16_000)
        XCTAssertGreaterThanOrEqual(wav.count, 44)
        XCTAssertEqual(String(data: wav[0..<4], encoding: .ascii), "RIFF")
        XCTAssertEqual(String(data: wav[8..<12], encoding: .ascii), "WAVE")
        XCTAssertEqual(String(data: wav[12..<16], encoding: .ascii), "fmt ")
        XCTAssertEqual(String(data: wav[36..<40], encoding: .ascii), "data")
    }

    func testEncodeWAV_correctSize() throws {
        let wav = try Gemma4Provider.encodeWAV(samples: Array(repeating: 0.25, count: 100), sampleRate: 16_000)
        XCTAssertEqual(wav.count, 44 + 200) // header + 100 samples * 2 bytes
    }

    func testEncodeWAV_clampsExtremeValues() throws {
        let wav = try Gemma4Provider.encodeWAV(samples: [5.0, -5.0], sampleRate: 16_000)
        let s0 = wav[44...45].withUnsafeBytes { $0.load(as: Int16.self).littleEndian }
        let s1 = wav[46...47].withUnsafeBytes { $0.load(as: Int16.self).littleEndian }
        XCTAssertEqual(s0, Int16.max)
        XCTAssertEqual(s1, -Int16.max)
    }

    func testEncodeWAV_emptyInput() throws {
        let wav = try Gemma4Provider.encodeWAV(samples: [], sampleRate: 16_000)
        XCTAssertEqual(wav.count, 44)
    }

    // MARK: - Response Parsing

    func testParseResponse_singlePart() throws {
        let json: [String: Any] = ["candidates": [["content": ["parts": [["text": "hello world"]]]]]]
        let data = try JSONSerialization.data(withJSONObject: json)
        XCTAssertEqual(try Gemma4Provider.parseResponse(data), "hello world")
    }

    func testParseResponse_multipleParts() throws {
        let json: [String: Any] = ["candidates": [["content": ["parts": [["text": "hello"], ["text": "world"]]]]]]
        let data = try JSONSerialization.data(withJSONObject: json)
        XCTAssertEqual(try Gemma4Provider.parseResponse(data), "hello world")
    }

    func testParseResponse_malformedJSON() {
        XCTAssertThrowsError(try Gemma4Provider.parseResponse("bad".data(using: .utf8)!))
    }

    func testParseResponse_missingCandidates() {
        let data = try! JSONSerialization.data(withJSONObject: ["error": "none"])
        XCTAssertThrowsError(try Gemma4Provider.parseResponse(data))
    }

    func testParseResponse_emptyParts() {
        let json: [String: Any] = ["candidates": [["content": ["parts": [] as [[String: Any]]]]]]
        let data = try! JSONSerialization.data(withJSONObject: json)
        XCTAssertThrowsError(try Gemma4Provider.parseResponse(data))
    }

    // MARK: - Request Body Building

    func testBuildRequestBody_structure() throws {
        let body = Gemma4Provider.buildRequestBody(base64Audio: "AAAA", prompt: "Transcribe")
        let contents = body["contents"] as? [[String: Any]]
        XCTAssertNotNil(contents)
        XCTAssertEqual(contents?.count, 1)

        let parts = (contents?.first?["parts"] as? [[String: Any]])
        XCTAssertEqual(parts?.count, 2)

        // First part: inline audio data
        let inlineData = parts?[0]["inline_data"] as? [String: Any]
        XCTAssertEqual(inlineData?["mime_type"] as? String, "audio/wav")
        XCTAssertEqual(inlineData?["data"] as? String, "AAAA")

        // Second part: text prompt
        XCTAssertEqual(parts?[1]["text"] as? String, "Transcribe")

        // Generation config
        let config = body["generationConfig"] as? [String: Any]
        XCTAssertEqual(config?["temperature"] as? Double, 0.0)
    }

    // MARK: - Provider State

    func testProviderName() {
        let provider = Gemma4Provider()
        XCTAssertEqual(provider.name, "Gemma 4")
        XCTAssertTrue(provider.isAvailable)
    }

    func testClearCache_doesNotThrow() async throws {
        try await Gemma4Provider().clearCache()
    }

    // MARK: - Audio Fixture Integration

    func testAudioFixture_encodesToValidWAV() throws {
        let samples = try AudioFixtureLoader.load16kMonoFloatSamples(named: "dictation_fixture", ext: "wav")
        XCTAssertGreaterThan(samples.count, 0)
        let wav = try Gemma4Provider.encodeWAV(samples: samples, sampleRate: 16_000)
        XCTAssertEqual(wav.count, 44 + samples.count * 2)
        XCTAssertEqual(String(data: wav[0..<4], encoding: .ascii), "RIFF")
    }

    // MARK: - SpeechModel Properties

    func testSpeechModel_gemma4_properties() {
        let m = SettingsStore.SpeechModel.gemma4
        XCTAssertEqual(m.displayName, "Gemma 4 (Audio)")
        XCTAssertEqual(m.languageSupport, "50+ Languages")
        XCTAssertFalse(m.requiresAppleSilicon)
        XCTAssertFalse(m.isWhisperModel)
        XCTAssertTrue(m.supportsStreaming)
        XCTAssertEqual(m.provider, .google)
        XCTAssertEqual(m.brandName, "Google")
        XCTAssertEqual(m.brandColorHex, "#4285F4")
        XCTAssertEqual(m.requiredMemoryGB, 1.0)
    }

    func testSpeechModel_googleProvider() {
        let models = SettingsStore.SpeechModel.allCases.filter { $0.provider == .google }
        XCTAssertTrue(models.contains(.gemma4))
        XCTAssertEqual(models.count, 1)
    }

    // MARK: - Error Descriptions

    func testErrors_haveDescriptions() {
        let errors: [Gemma4Provider.Gemma4Error] = [
            .missingEndpoint, .invalidAudioData, .emptyResponse,
            .requestFailed(statusCode: 500, body: "error"),
            .malformedResponse(detail: "bad"),
            .networkError(underlying: NSError(domain: "", code: 0)),
        ]
        for e in errors {
            XCTAssertNotNil(e.errorDescription)
            XCTAssertFalse(e.errorDescription!.isEmpty)
        }
    }
}
