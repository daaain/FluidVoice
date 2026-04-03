import Foundation
import XCTest

@testable import FluidVoice_Debug

@MainActor
final class GemmaE2BProviderTests: XCTestCase {

    // MARK: - WAV Encoding Tests

    func testEncodeWAV_headerIsValid() throws {
        let samples: [Float] = [0.0, 0.5, -0.5, 1.0, -1.0]
        let wav = try GemmaE2BProvider.encodeWAV(samples: samples, sampleRate: 16_000)

        // Minimum WAV header is 44 bytes
        XCTAssertGreaterThanOrEqual(wav.count, 44)

        // RIFF header
        let riff = String(data: wav[0..<4], encoding: .ascii)
        XCTAssertEqual(riff, "RIFF")

        // WAVE format
        let wave = String(data: wav[8..<12], encoding: .ascii)
        XCTAssertEqual(wave, "WAVE")

        // fmt chunk
        let fmt = String(data: wav[12..<16], encoding: .ascii)
        XCTAssertEqual(fmt, "fmt ")

        // data chunk
        let dataTag = String(data: wav[36..<40], encoding: .ascii)
        XCTAssertEqual(dataTag, "data")
    }

    func testEncodeWAV_dataSizeMatchesSamples() throws {
        let samples: [Float] = Array(repeating: 0.25, count: 100)
        let wav = try GemmaE2BProvider.encodeWAV(samples: samples, sampleRate: 16_000)

        // 44 byte header + 100 samples * 2 bytes each = 244
        XCTAssertEqual(wav.count, 244)

        // Verify data sub-chunk size field (bytes 40-43)
        let dataSize = wav.withUnsafeBytes { ptr -> UInt32 in
            ptr.load(fromByteOffset: 40, as: UInt32.self).littleEndian
        }
        XCTAssertEqual(dataSize, 200) // 100 * 2
    }

    func testEncodeWAV_clampingExtremeValues() throws {
        let samples: [Float] = [2.0, -3.0, 100.0, -100.0]
        let wav = try GemmaE2BProvider.encodeWAV(samples: samples, sampleRate: 16_000)

        // Extract 16-bit samples from the data section
        let int16Samples = wav[44...].withUnsafeBytes { ptr in
            (0..<4).map { i in
                ptr.load(fromByteOffset: i * 2, as: Int16.self).littleEndian
            }
        }

        // All should be clamped to Int16.max or Int16.min
        XCTAssertEqual(int16Samples[0], Int16.max) // 2.0 clamped to 1.0
        XCTAssertEqual(int16Samples[1], -Int16.max) // -3.0 clamped to -1.0
        XCTAssertEqual(int16Samples[2], Int16.max) // 100.0 clamped to 1.0
        XCTAssertEqual(int16Samples[3], -Int16.max) // -100.0 clamped to -1.0
    }

    func testEncodeWAV_roundTripSilence() throws {
        let samples: [Float] = Array(repeating: 0.0, count: 50)
        let wav = try GemmaE2BProvider.encodeWAV(samples: samples, sampleRate: 16_000)

        let int16Samples = wav[44...].withUnsafeBytes { ptr in
            (0..<50).map { i in
                ptr.load(fromByteOffset: i * 2, as: Int16.self).littleEndian
            }
        }

        for sample in int16Samples {
            XCTAssertEqual(sample, 0)
        }
    }

    func testEncodeWAV_emptyInput() throws {
        let samples: [Float] = []
        let wav = try GemmaE2BProvider.encodeWAV(samples: samples, sampleRate: 16_000)
        XCTAssertEqual(wav.count, 44) // Header only
    }

    // MARK: - Response Parsing Tests

    func testParseGeminiResponse_validSinglePart() throws {
        let json: [String: Any] = [
            "candidates": [
                [
                    "content": [
                        "parts": [
                            ["text": "Hello world, this is a test."],
                        ],
                    ],
                ],
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: json)
        let result = try GemmaE2BProvider.parseGeminiResponse(data)
        XCTAssertEqual(result, "Hello world, this is a test.")
    }

    func testParseGeminiResponse_multiPart() throws {
        let json: [String: Any] = [
            "candidates": [
                [
                    "content": [
                        "parts": [
                            ["text": "Hello"],
                            ["text": "world"],
                        ],
                    ],
                ],
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: json)
        let result = try GemmaE2BProvider.parseGeminiResponse(data)
        XCTAssertEqual(result, "Hello world")
    }

    func testParseGeminiResponse_emptyParts() {
        let json: [String: Any] = [
            "candidates": [
                [
                    "content": [
                        "parts": [] as [[String: Any]],
                    ],
                ],
            ],
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: json) else {
            XCTFail("Failed to serialize test JSON")
            return
        }
        XCTAssertThrowsError(try GemmaE2BProvider.parseGeminiResponse(data)) { error in
            guard let gemmaError = error as? GemmaE2BProvider.GemmaError else {
                XCTFail("Expected GemmaError, got \(error)")
                return
            }
            if case .malformedResponse = gemmaError {
                // Expected
            } else {
                XCTFail("Expected malformedResponse, got \(gemmaError)")
            }
        }
    }

    func testParseGeminiResponse_malformedJSON() {
        let data = "not json".data(using: .utf8)!
        XCTAssertThrowsError(try GemmaE2BProvider.parseGeminiResponse(data)) { error in
            guard let gemmaError = error as? GemmaE2BProvider.GemmaError else {
                XCTFail("Expected GemmaError")
                return
            }
            if case .malformedResponse = gemmaError {
                // Expected
            } else {
                XCTFail("Expected malformedResponse, got \(gemmaError)")
            }
        }
    }

    func testParseGeminiResponse_missingCandidates() {
        let json: [String: Any] = ["result": "no candidates"]
        guard let data = try? JSONSerialization.data(withJSONObject: json) else {
            XCTFail("Failed to serialize test JSON")
            return
        }
        XCTAssertThrowsError(try GemmaE2BProvider.parseGeminiResponse(data)) { error in
            guard let gemmaError = error as? GemmaE2BProvider.GemmaError,
                  case .malformedResponse = gemmaError else {
                XCTFail("Expected malformedResponse GemmaError")
                return
            }
        }
    }

    func testParseGeminiResponse_noTextParts() {
        let json: [String: Any] = [
            "candidates": [
                [
                    "content": [
                        "parts": [
                            ["inline_data": ["mime_type": "audio/wav"]],
                        ],
                    ],
                ],
            ],
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: json) else {
            XCTFail("Failed to serialize test JSON")
            return
        }
        XCTAssertThrowsError(try GemmaE2BProvider.parseGeminiResponse(data)) { error in
            guard let gemmaError = error as? GemmaE2BProvider.GemmaError,
                  case .malformedResponse = gemmaError else {
                XCTFail("Expected malformedResponse GemmaError")
                return
            }
        }
    }

    // MARK: - Provider State Tests

    func testProviderNotReady_withoutAPIKey() {
        let provider = GemmaE2BProvider()
        XCTAssertEqual(provider.name, "Gemma 4 E2B")
        XCTAssertTrue(provider.isAvailable)
    }

    func testProviderFillerFlag_default() {
        let provider = GemmaE2BProvider()
        XCTAssertFalse(provider.removeFillerWordsInline)
    }

    func testProviderFillerFlag_enabled() {
        let provider = GemmaE2BProvider(removeFillerWordsInline: true)
        XCTAssertTrue(provider.removeFillerWordsInline)
    }

    func testClearCache_doesNotThrow() async {
        let provider = GemmaE2BProvider()
        do {
            try await provider.clearCache()
        } catch {
            XCTFail("clearCache should not throw for cloud provider: \(error)")
        }
    }

    // MARK: - Audio Fixture Integration

    func testAudioFixture_encodesToValidWAV() throws {
        let samples = try AudioFixtureLoader.load16kMonoFloatSamples(named: "dictation_fixture", ext: "wav")
        XCTAssertGreaterThan(samples.count, 0)

        let wav = try GemmaE2BProvider.encodeWAV(samples: samples, sampleRate: 16_000)
        XCTAssertGreaterThan(wav.count, 44)

        // Verify the header
        let riff = String(data: wav[0..<4], encoding: .ascii)
        XCTAssertEqual(riff, "RIFF")

        // Verify data size matches
        let expectedDataBytes = samples.count * 2
        XCTAssertEqual(wav.count, 44 + expectedDataBytes)
    }

    // MARK: - SpeechModel Property Validation

    func testSpeechModel_gemma4E2B_properties() {
        let model = SettingsStore.SpeechModel.gemma4E2B

        XCTAssertEqual(model.displayName, "Gemma 4 E2B (Audio)")
        XCTAssertEqual(model.languageSupport, "50+ Languages")
        XCTAssertEqual(model.downloadSize, "Cloud API (No Download)")
        XCTAssertFalse(model.requiresAppleSilicon)
        XCTAssertFalse(model.isWhisperModel)
        XCTAssertNil(model.whisperModelFile)
        XCTAssertNil(model.whisperModelName)
        XCTAssertFalse(model.requiresMacOS26)
        XCTAssertFalse(model.requiresMacOS15)
        XCTAssertEqual(model.humanReadableName, "Gemma 4 E2B - Multimodal Audio")
        XCTAssertEqual(model.cardDescription, "Google multimodal model with native audio. Cloud API with built-in filler word removal.")
        XCTAssertEqual(model.requiredMemoryGB, 1.0)
        XCTAssertNil(model.memoryWarning)
        XCTAssertEqual(model.speedRating, 2)
        XCTAssertEqual(model.accuracyRating, 5)
        XCTAssertEqual(model.speedPercent, 0.35)
        XCTAssertEqual(model.accuracyPercent, 0.92)
        XCTAssertEqual(model.badgeText, "New")
        XCTAssertFalse(model.appleSiliconOptimized)
        XCTAssertFalse(model.supportsStreaming)
        XCTAssertEqual(model.provider, .google)
        XCTAssertEqual(model.brandName, "Google")
        XCTAssertFalse(model.usesAppleLogo)
        XCTAssertEqual(model.brandColorHex, "#4285F4")
    }

    func testSpeechModel_googleProvider_exists() {
        let googleModels = SettingsStore.SpeechModel.allCases.filter { $0.provider == .google }
        XCTAssertTrue(googleModels.contains(.gemma4E2B))
        XCTAssertEqual(googleModels.count, 1)
    }

    // MARK: - Error Description Tests

    func testGemmaError_missingAPIKey_description() {
        let error = GemmaE2BProvider.GemmaError.missingAPIKey
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription!.contains("API key"))
    }

    func testGemmaError_requestFailed_description() {
        let error = GemmaE2BProvider.GemmaError.requestFailed(statusCode: 401, message: "Unauthorized")
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription!.contains("401"))
        XCTAssertTrue(error.errorDescription!.contains("Unauthorized"))
    }

    func testGemmaError_emptyResponse_description() {
        let error = GemmaE2BProvider.GemmaError.emptyResponse
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription!.contains("empty"))
    }

    func testGemmaError_malformedResponse_description() {
        let error = GemmaE2BProvider.GemmaError.malformedResponse(detail: "missing field")
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription!.contains("missing field"))
    }

    func testGemmaError_networkError_description() {
        let underlying = NSError(domain: "NSURLErrorDomain", code: -1009, userInfo: [
            NSLocalizedDescriptionKey: "The Internet connection appears to be offline.",
        ])
        let error = GemmaE2BProvider.GemmaError.networkError(underlying: underlying)
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription!.contains("Network error"))
    }

    func testGemmaError_invalidAudioData_description() {
        let error = GemmaE2BProvider.GemmaError.invalidAudioData
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription!.contains("WAV"))
    }
}
