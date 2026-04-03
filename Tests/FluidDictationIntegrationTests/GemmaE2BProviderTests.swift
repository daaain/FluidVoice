import Foundation
import XCTest

@testable import FluidVoice_Debug

@MainActor
final class GemmaE2BProviderTests: XCTestCase {

    func testEncodeWAV_headerIsValid() throws {
        let samples: [Float] = [0.0, 0.5, -0.5, 1.0, -1.0]
        let wav = try GemmaE2BProvider.encodeWAV(samples: samples, sampleRate: 16_000)
        XCTAssertGreaterThanOrEqual(wav.count, 44)
        XCTAssertEqual(String(data: wav[0..<4], encoding: .ascii), "RIFF")
        XCTAssertEqual(String(data: wav[8..<12], encoding: .ascii), "WAVE")
        XCTAssertEqual(String(data: wav[12..<16], encoding: .ascii), "fmt ")
        XCTAssertEqual(String(data: wav[36..<40], encoding: .ascii), "data")
    }

    func testEncodeWAV_dataSizeMatchesSamples() throws {
        let samples: [Float] = Array(repeating: 0.25, count: 100)
        let wav = try GemmaE2BProvider.encodeWAV(samples: samples, sampleRate: 16_000)
        XCTAssertEqual(wav.count, 244)
        let dataSize = wav.withUnsafeBytes { ptr -> UInt32 in
            ptr.load(fromByteOffset: 40, as: UInt32.self).littleEndian
        }
        XCTAssertEqual(dataSize, 200)
    }

    func testEncodeWAV_clampingExtremeValues() throws {
        let samples: [Float] = [2.0, -3.0, 100.0, -100.0]
        let wav = try GemmaE2BProvider.encodeWAV(samples: samples, sampleRate: 16_000)
        let int16Samples = wav[44...].withUnsafeBytes { ptr in
            (0..<4).map { i in ptr.load(fromByteOffset: i * 2, as: Int16.self).littleEndian }
        }
        XCTAssertEqual(int16Samples[0], Int16.max)
        XCTAssertEqual(int16Samples[1], -Int16.max)
        XCTAssertEqual(int16Samples[2], Int16.max)
        XCTAssertEqual(int16Samples[3], -Int16.max)
    }

    func testEncodeWAV_emptyInput() throws {
        let wav = try GemmaE2BProvider.encodeWAV(samples: [], sampleRate: 16_000)
        XCTAssertEqual(wav.count, 44)
    }

    func testParseGeminiResponse_validSinglePart() throws {
        let json: [String: Any] = ["candidates": [["content": ["parts": [["text": "Hello world"]]]]]]
        let data = try JSONSerialization.data(withJSONObject: json)
        XCTAssertEqual(try GemmaE2BProvider.parseGeminiResponse(data), "Hello world")
    }

    func testParseGeminiResponse_multiPart() throws {
        let json: [String: Any] = ["candidates": [["content": ["parts": [["text": "Hello"], ["text": "world"]]]]]]
        let data = try JSONSerialization.data(withJSONObject: json)
        XCTAssertEqual(try GemmaE2BProvider.parseGeminiResponse(data), "Hello world")
    }

    func testParseGeminiResponse_malformedJSON() {
        XCTAssertThrowsError(try GemmaE2BProvider.parseGeminiResponse("not json".data(using: .utf8)!))
    }

    func testParseGeminiResponse_missingCandidates() {
        let data = try! JSONSerialization.data(withJSONObject: ["result": "none"])
        XCTAssertThrowsError(try GemmaE2BProvider.parseGeminiResponse(data))
    }

    func testProviderState() {
        let provider = GemmaE2BProvider()
        XCTAssertEqual(provider.name, "Gemma 4 E2B")
        XCTAssertTrue(provider.isAvailable)
        XCTAssertFalse(provider.removeFillerWordsInline)
    }

    func testProviderFillerFlag() {
        let provider = GemmaE2BProvider(removeFillerWordsInline: true)
        XCTAssertTrue(provider.removeFillerWordsInline)
    }

    func testAudioFixture_encodesToValidWAV() throws {
        let samples = try AudioFixtureLoader.load16kMonoFloatSamples(named: "dictation_fixture", ext: "wav")
        let wav = try GemmaE2BProvider.encodeWAV(samples: samples, sampleRate: 16_000)
        XCTAssertEqual(wav.count, 44 + samples.count * 2)
        XCTAssertEqual(String(data: wav[0..<4], encoding: .ascii), "RIFF")
    }

    func testSpeechModel_gemma4E2B_properties() {
        let model = SettingsStore.SpeechModel.gemma4E2B
        XCTAssertEqual(model.displayName, "Gemma 4 E2B (Audio)")
        XCTAssertEqual(model.languageSupport, "50+ Languages")
        XCTAssertEqual(model.downloadSize, "Cloud API (No Download)")
        XCTAssertFalse(model.requiresAppleSilicon)
        XCTAssertFalse(model.isWhisperModel)
        XCTAssertEqual(model.provider, .google)
        XCTAssertEqual(model.brandName, "Google")
        XCTAssertEqual(model.brandColorHex, "#4285F4")
        XCTAssertFalse(model.supportsStreaming)
        XCTAssertEqual(model.requiredMemoryGB, 1.0)
    }

    func testSpeechModel_googleProvider() {
        let models = SettingsStore.SpeechModel.allCases.filter { $0.provider == .google }
        XCTAssertEqual(models.count, 1)
        XCTAssertTrue(models.contains(.gemma4E2B))
    }

    func testGemmaErrors_haveDescriptions() {
        let errors: [GemmaE2BProvider.GemmaError] = [
            .missingAPIKey, .invalidAudioData, .emptyResponse,
            .requestFailed(statusCode: 401, message: "test"),
            .malformedResponse(detail: "test"),
            .networkError(underlying: NSError(domain: "", code: 0)),
        ]
        for error in errors {
            XCTAssertNotNil(error.errorDescription)
        }
    }
}
