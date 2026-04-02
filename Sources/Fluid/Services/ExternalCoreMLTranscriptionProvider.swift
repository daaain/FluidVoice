import Foundation

#if arch(arm64)
import FluidAudio

@available(macOS 15.0, *)
final class ExternalCoreMLTranscriptionProvider: TranscriptionProvider {
    let name = "External CoreML"

    var isAvailable: Bool { true }
    private(set) var isReady: Bool = false
    var prefersNativeFileTranscription: Bool { true }
    private let streamingPreviewMaxSeconds: Double = 12

    private var cohereManager: CohereTranscribeAsrManager?
    private let modelOverride: SettingsStore.SpeechModel?
    private var loadedManifest: ExternalCoreMLManifestIdentity?

    init(modelOverride: SettingsStore.SpeechModel? = nil) {
        self.modelOverride = modelOverride
    }

    func prepare(progressHandler: ((Double) -> Void)? = nil) async throws {
        guard self.isReady == false else { return }

        let model = self.modelOverride ?? SettingsStore.shared.selectedSpeechModel
        DebugLogger.shared.info(
            "ExternalCoreML: prepare requested for model=\(model.rawValue)",
            source: "ExternalCoreML"
        )
        guard let spec = model.externalCoreMLSpec else {
            DebugLogger.shared.error(
                "ExternalCoreML: missing spec for model=\(model.rawValue)",
                source: "ExternalCoreML"
            )
            throw Self.makeError("No external CoreML spec registered for \(model.displayName).")
        }
        guard let directory = Self.artifactsDirectory(for: model, spec: spec) else {
            DebugLogger.shared.error(
                "ExternalCoreML: unable to resolve cache directory for model=\(model.rawValue)",
                source: "ExternalCoreML"
            )
            throw Self.makeError("Unable to resolve a cache directory for \(model.displayName).")
        }

        try await self.ensureArtifactsPresent(
            for: model,
            spec: spec,
            at: directory,
            progressHandler: progressHandler
        )

        progressHandler?(0.85)

        self.loadedManifest = try spec.loadManifest(at: directory)

        switch spec.backend {
        case .cohereTranscribe:
            let manager = CohereTranscribeAsrManager()
            progressHandler?(0.9)
            try self.invalidateCompiledCohereCacheIfNeeded(at: directory)
            let computeSummary = [
                String(describing: spec.computeConfiguration.frontend),
                String(describing: spec.computeConfiguration.encoder),
                String(describing: spec.computeConfiguration.crossKV),
                String(describing: spec.computeConfiguration.decoder),
            ].joined(separator: "/")
            DebugLogger.shared.info(
                "ExternalCoreML: loading Cohere models [splitCompute=\(computeSummary), maxAudioSamples=\(self.loadedManifest?.maxAudioSamples ?? 0)]",
                source: "ExternalCoreML"
            )
            try await manager.loadModels(from: directory, computeConfiguration: spec.computeConfiguration)
            self.cohereManager = manager
            self.persistCompiledCohereCacheStamp(at: directory)
        }

        self.isReady = true
        DebugLogger.shared.info(
            "ExternalCoreML: provider ready for model=\(model.rawValue)",
            source: "ExternalCoreML"
        )
        progressHandler?(1.0)
    }

    func transcribe(_ samples: [Float]) async throws -> ASRTranscriptionResult {
        try await self.transcribeFinal(samples)
    }

    func transcribeStreaming(_ samples: [Float]) async throws -> ASRTranscriptionResult {
        let previewSamples = self.previewSamples(for: samples)
        DebugLogger.shared.debug(
            "ExternalCoreML: streaming preview request [samples=\(samples.count), previewSamples=\(previewSamples.count)]",
            source: "ExternalCoreML"
        )
        guard let manager = self.cohereManager else {
            DebugLogger.shared.error(
                "ExternalCoreML: streaming preview requested before manager initialization",
                source: "ExternalCoreML"
            )
            throw Self.makeError("External CoreML model is not initialized.")
        }

        let text = try await manager.transcribe(audioSamples: self.paddedSamplesToModelLimit(previewSamples))
        return ASRTranscriptionResult(text: text, confidence: 1.0)
    }

    func transcribeFile(at fileURL: URL) async throws -> ASRTranscriptionResult {
        guard let manager = self.cohereManager else {
            DebugLogger.shared.error(
                "ExternalCoreML: file transcription requested before manager initialization",
                source: "ExternalCoreML"
            )
            throw Self.makeError("External CoreML model is not initialized.")
        }

        let startedAt = Date()
        DebugLogger.shared.info(
            "ExternalCoreML: native file transcription start [file=\(fileURL.lastPathComponent)]",
            source: "ExternalCoreML"
        )
        let text = try await manager.transcribe(audioFileAt: fileURL)
        let elapsed = Date().timeIntervalSince(startedAt)
        DebugLogger.shared.info(
            "ExternalCoreML: native file transcription finished in \(String(format: "%.2f", elapsed))s [chars=\(text.count)]",
            source: "ExternalCoreML"
        )
        return ASRTranscriptionResult(text: text, confidence: 1.0)
    }

    func transcribeFinal(_ samples: [Float]) async throws -> ASRTranscriptionResult {
        guard let manager = self.cohereManager else {
            DebugLogger.shared.error(
                "ExternalCoreML: transcribe requested before manager initialization",
                source: "ExternalCoreML"
            )
            throw Self.makeError("External CoreML model is not initialized.")
        }
        let startedAt = Date()
        let sampleRate = Double((self.modelOverride ?? SettingsStore.shared.selectedSpeechModel).externalCoreMLSpec?.expectedSampleRate ?? 16_000)
        let audioSeconds = sampleRate > 0 ? Double(samples.count) / sampleRate : 0
        DebugLogger.shared.debug(
            "ExternalCoreML: transcribing \(samples.count) samples [audioSeconds=\(String(format: "%.2f", audioSeconds))]",
            source: "ExternalCoreML"
        )
        let text = try await self.transcribeByManifestWindow(samples, manager: manager)
        let elapsed = Date().timeIntervalSince(startedAt)
        let rtf = audioSeconds > 0 ? elapsed / audioSeconds : 0
        DebugLogger.shared.info(
            "ExternalCoreML: transcription finished in \(String(format: "%.2f", elapsed))s [audioSeconds=\(String(format: "%.2f", audioSeconds)), rtf=\(String(format: "%.2fx", rtf)), chars=\(text.count)]",
            source: "ExternalCoreML"
        )
        return ASRTranscriptionResult(text: text, confidence: 1.0)
    }

    func modelsExistOnDisk() -> Bool {
        let model = self.modelOverride ?? SettingsStore.shared.selectedSpeechModel
        guard let spec = model.externalCoreMLSpec,
              let directory = Self.artifactsDirectory(for: model, spec: spec)
        else {
            return false
        }
        return spec.validateArtifacts(at: directory)
    }

    func clearCache() async throws {
        let model = self.modelOverride ?? SettingsStore.shared.selectedSpeechModel
        guard let spec = model.externalCoreMLSpec,
              let directory = Self.artifactsDirectory(for: model, spec: spec)
        else {
            self.isReady = false
            self.cohereManager = nil
            return
        }

        let compiledDirectory = CohereTranscribeAsrModels.compiledArtifactsDirectory(for: directory)

        if FileManager.default.fileExists(atPath: compiledDirectory.path) {
            DebugLogger.shared.info(
                "ExternalCoreML: clearing compiled cache at \(compiledDirectory.path)",
                source: "ExternalCoreML"
            )
            try FileManager.default.removeItem(at: compiledDirectory)
        }

        if FileManager.default.fileExists(atPath: directory.path), Self.isAppManagedArtifactsDirectory(directory, spec: spec) {
            DebugLogger.shared.info(
                "ExternalCoreML: removing downloaded artifacts at \(directory.path)",
                source: "ExternalCoreML"
            )
            try FileManager.default.removeItem(at: directory)
        } else if FileManager.default.fileExists(atPath: directory.path) {
            DebugLogger.shared.warning(
                "ExternalCoreML: skipping deletion for non-managed artifacts directory at \(directory.path)",
                source: "ExternalCoreML"
            )
        }

        self.isReady = false
        self.cohereManager = nil
        self.loadedManifest = nil
        DebugLogger.shared.info(
            "ExternalCoreML: provider reset after cache clear",
            source: "ExternalCoreML"
        )
    }

    private func ensureArtifactsPresent(
        for model: SettingsStore.SpeechModel,
        spec: ExternalCoreMLASRModelSpec,
        at directory: URL,
        progressHandler: ((Double) -> Void)?
    ) async throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        if spec.validateArtifacts(at: directory) {
            DebugLogger.shared.info(
                "ExternalCoreML: artifact validation passed for \(directory.lastPathComponent)",
                source: "ExternalCoreML"
            )
            progressHandler?(0.8)
            return
        }

        guard let owner = spec.repositoryOwner, let repo = spec.repositoryName else {
            throw Self.makeError("Missing repository metadata for \(model.displayName).")
        }

        DebugLogger.shared.info(
            "ExternalCoreML: downloading missing artifacts from \(owner)/\(repo)",
            source: "ExternalCoreML"
        )

        let downloader = HuggingFaceModelDownloader(
            owner: owner,
            repo: repo,
            revision: spec.repositoryRevision,
            requiredItems: spec.requiredEntries.map { .init(path: $0, isDirectory: $0.hasSuffix(".mlpackage")) }
        )
        try await downloader.ensureModelsPresent(at: directory) { progress, item in
            DebugLogger.shared.debug(
                "ExternalCoreML: download progress \(Int(progress * 100))% [\(item)]",
                source: "ExternalCoreML"
            )
            progressHandler?(progress * 0.8)
        }

        do {
            try spec.validateArtifactsOrThrow(at: directory)
        } catch {
            throw Self.makeError(error.localizedDescription)
        }

        SettingsStore.shared.setExternalCoreMLArtifactsDirectory(directory, for: model)
    }

    private static func artifactsDirectory(
        for model: SettingsStore.SpeechModel,
        spec: ExternalCoreMLASRModelSpec
    ) -> URL? {
        SettingsStore.shared.externalCoreMLArtifactsDirectory(for: model) ?? spec.defaultCacheDirectory
    }

    private static func isAppManagedArtifactsDirectory(
        _ directory: URL,
        spec: ExternalCoreMLASRModelSpec
    ) -> Bool {
        guard let defaultCacheDirectory = spec.defaultCacheDirectory else { return false }
        return directory.standardizedFileURL.path == defaultCacheDirectory.standardizedFileURL.path
    }

    private static func makeError(_ description: String) -> NSError {
        NSError(
            domain: "ExternalCoreMLTranscriptionProvider",
            code: -1,
            userInfo: [NSLocalizedDescriptionKey: description]
        )
    }

    private func invalidateCompiledCohereCacheIfNeeded(at directory: URL) throws {
        guard let manifest = self.loadedManifest else { return }

        let compiledDirectory = CohereTranscribeAsrModels.compiledArtifactsDirectory(for: directory)
        guard FileManager.default.fileExists(atPath: compiledDirectory.path) else { return }

        let currentStamp = Self.compiledCohereCacheStamp(for: manifest)
        let stampURL = Self.compiledCohereCacheStampURL(for: directory)
        let previousStamp = try? String(contentsOf: stampURL, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        guard previousStamp != currentStamp else { return }

        let reason = previousStamp == nil ? "missing cache stamp" : "manifest changed"
        DebugLogger.shared.warning(
            "ExternalCoreML: clearing stale compiled Cohere cache [reason=\(reason)]",
            source: "ExternalCoreML"
        )
        try FileManager.default.removeItem(at: compiledDirectory)
    }

    private func persistCompiledCohereCacheStamp(at directory: URL) {
        guard let manifest = self.loadedManifest else { return }
        let stampURL = Self.compiledCohereCacheStampURL(for: directory)
        let stamp = Self.compiledCohereCacheStamp(for: manifest)
        try? stamp.write(to: stampURL, atomically: true, encoding: .utf8)
    }

    private static func compiledCohereCacheStampURL(for directory: URL) -> URL {
        directory.appendingPathComponent(".cohere_compiled_cache_stamp", isDirectory: false)
    }

    private static func compiledCohereCacheStamp(for manifest: ExternalCoreMLManifestIdentity) -> String {
        [
            manifest.modelID,
            String(manifest.sampleRate),
            String(manifest.maxAudioSamples),
            String(manifest.maxAudioSeconds),
            String(manifest.overlapSamples ?? 0),
        ].joined(separator: "|")
    }

    private func previewSamples(for samples: [Float]) -> [Float] {
        let sampleRate = self.loadedManifest?.sampleRate
            ?? (self.modelOverride ?? SettingsStore.shared.selectedSpeechModel).externalCoreMLSpec?.expectedSampleRate
            ?? 16_000
        let maxPreviewSamples = Int(Double(sampleRate) * self.streamingPreviewMaxSeconds)
        guard samples.count > maxPreviewSamples else { return samples }
        return Array(samples.suffix(maxPreviewSamples))
    }

    private func transcribeByManifestWindow(
        _ samples: [Float],
        manager: CohereTranscribeAsrManager
    ) async throws -> String {
        let maxAudioSamples = self.loadedManifest?.maxAudioSamples ?? 0
        guard maxAudioSamples > 0 else {
            return try await manager.transcribe(audioSamples: samples)
        }

        if samples.count <= maxAudioSamples {
            return try await manager.transcribe(audioSamples: self.paddedSamplesToModelLimit(samples))
        }

        let overlapSamples = min(self.loadedManifest?.overlapSamples ?? 0, maxAudioSamples / 2)
        let step = max(maxAudioSamples - overlapSamples, 1)
        var chunkTexts: [String] = []
        var startIndex = 0

        while startIndex < samples.count {
            let endIndex = min(startIndex + maxAudioSamples, samples.count)
            let chunk = Array(samples[startIndex..<endIndex])
            let text = try await manager.transcribe(audioSamples: self.paddedSamplesToModelLimit(chunk))
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty == false {
                chunkTexts.append(trimmed)
            }
            if endIndex >= samples.count {
                break
            }
            startIndex += step
        }

        return chunkTexts.joined(separator: " ")
    }

    private func paddedSamplesToModelLimit(_ samples: [Float]) -> [Float] {
        let maxAudioSamples = self.loadedManifest?.maxAudioSamples ?? samples.count
        guard maxAudioSamples > 0 else { return samples }

        if samples.count == maxAudioSamples {
            return samples
        }

        if samples.count > maxAudioSamples {
            return Array(samples.suffix(maxAudioSamples))
        }

        return samples + Array(repeating: 0, count: maxAudioSamples - samples.count)
    }
}

#else

final class ExternalCoreMLTranscriptionProvider: TranscriptionProvider {
    let name = "External CoreML"
    let isAvailable = false
    let isReady = false

    init(modelOverride: SettingsStore.SpeechModel? = nil) {}

    func prepare(progressHandler: ((Double) -> Void)? = nil) async throws {
        throw NSError(
            domain: "ExternalCoreMLTranscriptionProvider",
            code: -1,
            userInfo: [NSLocalizedDescriptionKey: "External CoreML models are only supported on Apple Silicon Macs."]
        )
    }

    func transcribe(_ samples: [Float]) async throws -> ASRTranscriptionResult {
        throw NSError(
            domain: "ExternalCoreMLTranscriptionProvider",
            code: -1,
            userInfo: [NSLocalizedDescriptionKey: "External CoreML models are only supported on Apple Silicon Macs."]
        )
    }
}

#endif
