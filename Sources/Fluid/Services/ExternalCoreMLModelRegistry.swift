import CoreML
import FluidAudio
import Foundation

enum ExternalCoreMLASRBackend {
    case cohereTranscribe
}

struct ExternalCoreMLManifestIdentity: Decodable {
    let modelID: String
    let sampleRate: Int
    let maxAudioSamples: Int
    let maxAudioSeconds: Double
    let overlapSamples: Int?

    private enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case sampleRate = "sample_rate"
        case maxAudioSamples = "max_audio_samples"
        case maxAudioSeconds = "max_audio_seconds"
        case overlapSamples = "overlap_samples"
    }
}

enum ExternalCoreMLArtifactsValidationError: LocalizedError {
    case missingEntries([String])
    case manifestMissing(URL)
    case manifestUnreadable(URL, Error)
    case unexpectedModelID(expected: String, actual: String)
    case unexpectedSampleRate(expected: Int, actual: Int)
    case unexpectedMaxAudioSamples(expected: Int, actual: Int)
    case unexpectedMaxAudioSeconds(expected: Double, actual: Double)

    var errorDescription: String? {
        switch self {
        case let .missingEntries(entries):
            return "Missing required files: \(entries.joined(separator: ", "))"
        case let .manifestMissing(url):
            return "Manifest file not found at \(url.path)"
        case let .manifestUnreadable(url, error):
            return "Failed to read manifest at \(url.path): \(error.localizedDescription)"
        case let .unexpectedModelID(expected, actual):
            return "Unexpected model_id '\(actual)'. Expected '\(expected)'."
        case let .unexpectedSampleRate(expected, actual):
            return "Unexpected sample rate \(actual). Expected \(expected)."
        case let .unexpectedMaxAudioSamples(expected, actual):
            return "Unexpected max audio samples \(actual). Expected \(expected)."
        case let .unexpectedMaxAudioSeconds(expected, actual):
            return "Unexpected max audio seconds \(actual). Expected \(expected)."
        }
    }
}

struct ExternalCoreMLASRModelSpec {
    let backend: ExternalCoreMLASRBackend
    let artifactFolderHint: String
    let manifestFileName: String
    let frontendFileName: String
    let encoderFileName: String
    let crossKVProjectorFileName: String?
    let decoderFileName: String
    let cachedDecoderFileName: String
    let expectedModelID: String
    let expectedSampleRate: Int
    let expectedMaxAudioSamples: Int
    let expectedMaxAudioSeconds: Double
    let computeConfiguration: CohereTranscribeComputeConfiguration
    let sourceURL: URL?
    let repositoryOwner: String?
    let repositoryName: String?
    let repositoryRevision: String

    var requiredEntries: [String] {
        [
            self.manifestFileName,
            self.frontendFileName,
            self.encoderFileName,
            self.crossKVProjectorFileName,
            self.decoderFileName,
            self.cachedDecoderFileName,
        ]
        .compactMap { $0 }
    }

    func url(for entry: String, in directory: URL) -> URL {
        directory.appendingPathComponent(entry, isDirectory: entry.hasSuffix(".mlpackage"))
    }

    var defaultCacheDirectory: URL? {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first?
            .appendingPathComponent(self.artifactFolderHint, isDirectory: true)
    }

    func validateArtifacts(at directory: URL) -> Bool {
        (try? self.validateArtifactsOrThrow(at: directory)) != nil
    }

    func missingEntries(at directory: URL) -> [String] {
        self.requiredEntries.filter { entry in
            let url = self.url(for: entry, in: directory)
            return FileManager.default.fileExists(atPath: url.path) == false
        }
    }

    func loadManifest(at directory: URL) throws -> ExternalCoreMLManifestIdentity {
        let manifestURL = self.url(for: self.manifestFileName, in: directory)
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw ExternalCoreMLArtifactsValidationError.manifestMissing(manifestURL)
        }

        do {
            let data = try Data(contentsOf: manifestURL)
            return try JSONDecoder().decode(ExternalCoreMLManifestIdentity.self, from: data)
        } catch {
            throw ExternalCoreMLArtifactsValidationError.manifestUnreadable(manifestURL, error)
        }
    }

    func validateArtifactsOrThrow(at directory: URL) throws {
        let missingEntries = self.missingEntries(at: directory)
        guard missingEntries.isEmpty else {
            throw ExternalCoreMLArtifactsValidationError.missingEntries(missingEntries)
        }

        let manifest = try self.loadManifest(at: directory)

        guard manifest.modelID == self.expectedModelID else {
            throw ExternalCoreMLArtifactsValidationError.unexpectedModelID(
                expected: self.expectedModelID,
                actual: manifest.modelID
            )
        }

        guard manifest.sampleRate == self.expectedSampleRate else {
            throw ExternalCoreMLArtifactsValidationError.unexpectedSampleRate(
                expected: self.expectedSampleRate,
                actual: manifest.sampleRate
            )
        }

        guard manifest.maxAudioSamples == self.expectedMaxAudioSamples else {
            throw ExternalCoreMLArtifactsValidationError.unexpectedMaxAudioSamples(
                expected: self.expectedMaxAudioSamples,
                actual: manifest.maxAudioSamples
            )
        }

        guard manifest.maxAudioSeconds == self.expectedMaxAudioSeconds else {
            throw ExternalCoreMLArtifactsValidationError.unexpectedMaxAudioSeconds(
                expected: self.expectedMaxAudioSeconds,
                actual: manifest.maxAudioSeconds
            )
        }
    }
}

enum ExternalCoreMLModelRegistry {
    static func spec(for model: SettingsStore.SpeechModel) -> ExternalCoreMLASRModelSpec? {
        switch model {
        case .cohereTranscribeSixBit:
            return ExternalCoreMLASRModelSpec(
                backend: .cohereTranscribe,
                artifactFolderHint: "cohere-transcribe-03-2026-CoreML-6bit",
                manifestFileName: "coreml_manifest.json",
                frontendFileName: "cohere_frontend.mlpackage",
                encoderFileName: "cohere_encoder.mlpackage",
                crossKVProjectorFileName: "cohere_cross_kv_projector.mlpackage",
                decoderFileName: "cohere_decoder_fullseq_masked.mlpackage",
                cachedDecoderFileName: "cohere_decoder_cached.mlpackage",
                expectedModelID: "CohereLabs/cohere-transcribe-03-2026",
                expectedSampleRate: 16_000,
                expectedMaxAudioSamples: 560_000,
                expectedMaxAudioSeconds: 35.0,
                computeConfiguration: .aneSmall,
                sourceURL: URL(string: "https://huggingface.co/BarathwajAnandan/cohere-transcribe-03-2026-CoreML-6bit"),
                repositoryOwner: "BarathwajAnandan",
                repositoryName: "cohere-transcribe-03-2026-CoreML-6bit",
                repositoryRevision: "main"
            )
        default:
            return nil
        }
    }
}

extension SettingsStore.SpeechModel {
    var externalCoreMLSpec: ExternalCoreMLASRModelSpec? {
        ExternalCoreMLModelRegistry.spec(for: self)
    }

    var requiresExternalArtifacts: Bool {
        self.externalCoreMLSpec != nil
    }

    var supportsCustomVocabulary: Bool {
        switch self {
        case .parakeetTDT, .parakeetTDTv2:
            return true
        default:
            return false
        }
    }
}
