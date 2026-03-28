import AppKit
import Combine
import SwiftUI

@MainActor
final class VoiceEngineSettingsViewModel: ObservableObject {
    let settings: SettingsStore
    private let appServices: AppServices

    var asr: ASRService { self.appServices.asr }

    @Published var modelSortOption: ModelSortOption = .provider
    @Published var providerFilter: SpeechProviderFilter = .all
    @Published var englishOnlyFilter: Bool = false
    @Published var installedOnlyFilter: Bool = false
    @Published var showSpeechFilters: Bool = false

    @Published var selectedSpeechProvider: SettingsStore.SpeechModel.Provider
    @Published var previewSpeechModel: SettingsStore.SpeechModel
    @Published var showAdvancedSpeechInfo: Bool = false
    @Published var suppressSpeechProviderSync: Bool = false
    @Published var skipNextSpeechModelSync: Bool = false

    /// The model currently being downloaded (nil if no download in progress)
    @Published var downloadingModel: SettingsStore.SpeechModel?
    /// Download progress for the current model (0.0 to 1.0)
    @Published var downloadProgress: Double = 0.0

    @Published var removeFillerWordsEnabled: Bool

    init(settings: SettingsStore, appServices: AppServices) {
        self.settings = settings
        self.appServices = appServices
        self.previewSpeechModel = settings.selectedSpeechModel
        self.selectedSpeechProvider = settings.selectedSpeechModel.provider
        self.removeFillerWordsEnabled = settings.removeFillerWordsEnabled
    }

    func onAppear() {
        self.previewSpeechModel = self.settings.selectedSpeechModel
        self.selectedSpeechProvider = self.settings.selectedSpeechModel.provider
        self.removeFillerWordsEnabled = self.settings.removeFillerWordsEnabled

        Task {
            await self.asr.checkIfModelsExistAsync()
        }
    }

    func handleSelectedSpeechModelChange(_ newValue: SettingsStore.SpeechModel) {
        if self.skipNextSpeechModelSync {
            self.skipNextSpeechModelSync = false
            return
        }
        guard !self.suppressSpeechProviderSync else { return }
        self.previewSpeechModel = newValue
        self.setSelectedSpeechProvider(newValue.provider)
    }

    var filteredSpeechModels: [SettingsStore.SpeechModel] {
        var models = SettingsStore.SpeechModel.availableModels

        switch self.providerFilter {
        case .all:
            break
        case .nvidia:
            models = models.filter { $0.provider == .nvidia }
        case .apple:
            models = models.filter { $0.provider == .apple }
        case .cohere:
            models = models.filter { $0.provider == .cohere }
        case .openai:
            models = models.filter { $0.provider == .openai }
        }

        if self.englishOnlyFilter {
            models = models.filter { model in
                let label = model.languageSupport.lowercased()
                let title = model.humanReadableName.lowercased()
                return label.contains("english only") || title.contains("english")
            }
        }

        if self.installedOnlyFilter {
            models = models.filter { $0.isInstalled }
        }

        switch self.modelSortOption {
        case .provider:
            models.sort { $0.brandName.localizedCaseInsensitiveCompare($1.brandName) == .orderedAscending }
        case .accuracy:
            models.sort { $0.accuracyPercent > $1.accuracyPercent }
        case .speed:
            models.sort { $0.speedPercent > $1.speedPercent }
        }

        return models
    }

    func activateSpeechModel(_ model: SettingsStore.SpeechModel) {
        guard !self.asr.isRunning else { return }
        if model.requiresExternalArtifacts && self.ensureExternalArtifactsConfigured(for: model) == false {
            return
        }
        withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
            self.settings.selectedSpeechModel = model
            self.previewSpeechModel = model
            self.setSelectedSpeechProvider(model.provider)
        }
        self.asr.resetTranscriptionProvider()
        Task {
            do {
                try await self.asr.ensureAsrReady()
            } catch {
                DebugLogger.shared.error("Failed to prepare model after activation: \(error)", source: "AISettingsView")
                self.asr.errorTitle = "Model Activation Failed"
                self.asr.errorMessage = error.localizedDescription
                self.asr.showError = true
            }
        }
    }

    func downloadSpeechModel(_ model: SettingsStore.SpeechModel) {
        guard !self.asr.isRunning else { return }
        if model.requiresExternalArtifacts {
            _ = self.ensureExternalArtifactsConfigured(for: model)
            return
        }
        guard self.downloadingModel == nil else { return } // Prevent concurrent downloads
        self.downloadingModel = model
        self.downloadProgress = 0.0

        Task {
            // Mark this model as downloading
            self.downloadingModel = model
            self.downloadProgress = 0.0

            do {
                // Download the model WITHOUT changing the active selection
                // Capture the model ID to guard against stale progress callbacks
                let downloadingModelId = model.id
                try await self.asr.downloadModel(model, progressHandler: { [weak self] progress in
                    Task { @MainActor in
                        // Only update progress if we're still downloading this specific model
                        guard let self, self.downloadingModel?.id == downloadingModelId else { return }
                        self.downloadProgress = max(self.downloadProgress, progress)
                    }
                })

                DebugLogger.shared.info("Model download completed: \(model.displayName)", source: "VoiceEngineVM")
            } catch {
                DebugLogger.shared.error("Failed to download model \(model.displayName): \(error)", source: "VoiceEngineVM")
                await MainActor.run {
                    self.asr.errorTitle = model.requiresExternalArtifacts ? "Model Import Failed" : "Model Download Failed"
                    self.asr.errorMessage = error.localizedDescription
                    self.asr.showError = true
                }
            }

            // Clear downloading state
            self.downloadingModel = nil
            self.downloadProgress = 0.0
        }
    }

    func deleteSpeechModel(_ model: SettingsStore.SpeechModel) {
        guard !self.asr.isRunning else { return }
        let previousActive = self.settings.selectedSpeechModel

        Task {
            let shouldRestore = previousActive != model
            await MainActor.run {
                if shouldRestore {
                    self.suppressSpeechProviderSync = true
                }
                self.settings.selectedSpeechModel = model
                self.asr.resetTranscriptionProvider()
            }

            defer {
                Task { @MainActor in
                    guard shouldRestore else { return }
                    self.skipNextSpeechModelSync = true
                    self.settings.selectedSpeechModel = previousActive
                    self.asr.resetTranscriptionProvider()
                    if self.previewSpeechModel == model {
                        self.previewSpeechModel = model
                    }
                    self.suppressSpeechProviderSync = false
                }
            }

            await self.deleteModels()
        }
    }

    func isActiveSpeechModel(_ model: SettingsStore.SpeechModel) -> Bool {
        self.settings.selectedSpeechModel == model
    }

    var modelDescriptionText: String {
        let model = self.settings.selectedSpeechModel
        switch model {
        case .appleSpeech:
            return "Apple Speech (Legacy) uses on-device recognition. No download required, works on Intel and Apple Silicon."
        case .appleSpeechAnalyzer:
            return "Apple Speech uses advanced on-device recognition with fast, accurate transcription. Requires macOS 26+."
        case .parakeetTDT:
            return "Parakeet TDT v3 uses CoreML and Neural Engine for fastest transcription (25 languages) on Apple Silicon."
        case .parakeetTDTv2:
            return "Parakeet TDT v2 is an English-only model optimized for accuracy and consistency on Apple Silicon."
        case .qwen3Asr:
            return "Qwen3 ASR is a multilingual FluidAudio model with strong quality, but higher memory usage. Requires macOS 15+."
        case .cohereTranscribeSixBit:
            return "Cohere Transcribe uses an external CoreML pipeline loaded from a local folder. Best on Apple Silicon with 8GB+ RAM."
        default:
            return "Whisper models support 99 languages and work on any Mac."
        }
    }

    func downloadModels() async {
        do {
            try await self.asr.ensureAsrReady()
        } catch {
            DebugLogger.shared.error("Failed to download models: \(error)", source: "AISettingsView")
            self.asr.errorTitle = self.settings.selectedSpeechModel.requiresExternalArtifacts ? "Model Import Failed" : "Model Download Failed"
            self.asr.errorMessage = error.localizedDescription
            self.asr.showError = true
        }
    }

    func deleteModels() async {
        do {
            try await self.asr.clearModelCache()
            let model = self.settings.selectedSpeechModel
            if model.requiresExternalArtifacts {
                self.settings.setExternalCoreMLArtifactsDirectory(nil, for: model)
                self.asr.resetTranscriptionProvider()
            }
        } catch {
            DebugLogger.shared.error("Failed to delete models: \(error)", source: "AISettingsView")
        }
    }

    func setSelectedSpeechProvider(_ provider: SettingsStore.SpeechModel.Provider) {
        self.selectedSpeechProvider = provider
    }

    func externalArtifactsDirectoryDisplay(for model: SettingsStore.SpeechModel) -> String? {
        self.settings.externalCoreMLArtifactsDirectory(for: model)?.path
    }

    func chooseExternalArtifactsDirectory(for model: SettingsStore.SpeechModel) {
        _ = self.ensureExternalArtifactsConfigured(for: model, forceChooser: true)
    }

    func clearExternalArtifactsDirectory(for model: SettingsStore.SpeechModel) {
        self.settings.setExternalCoreMLArtifactsDirectory(nil, for: model)
        if self.settings.selectedSpeechModel == model {
            self.asr.resetTranscriptionProvider()
        }
    }

    func openExternalModelSource(for model: SettingsStore.SpeechModel) {
        guard let url = model.externalCoreMLSpec?.sourceURL else { return }
        NSWorkspace.shared.open(url)
    }

    @discardableResult
    private func ensureExternalArtifactsConfigured(
        for model: SettingsStore.SpeechModel,
        forceChooser: Bool = false
    ) -> Bool {
        guard let spec = model.externalCoreMLSpec else { return true }
        if forceChooser == false,
           let directory = self.settings.externalCoreMLArtifactsDirectory(for: model),
           spec.validateArtifacts(at: directory)
        {
            return true
        }

        let panel = NSOpenPanel()
        panel.title = "Select \(model.displayName) Artifacts Folder"
        panel.message = "Choose the folder containing \(spec.artifactFolderHint) and its CoreML files."
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = false
        panel.prompt = "Use Folder"

        guard panel.runModal() == .OK, let selectedDirectory = panel.url?.standardizedFileURL else {
            return false
        }

        let candidateDirectory = spec.validateArtifacts(at: selectedDirectory)
            ? selectedDirectory
            : selectedDirectory.appendingPathComponent(spec.artifactFolderHint, isDirectory: true)

        do {
            try spec.validateArtifactsOrThrow(at: candidateDirectory)
        } catch {
            self.asr.errorTitle = "Artifacts Folder Invalid"
            self.asr.errorMessage = error.localizedDescription
            self.asr.showError = true
            return false
        }

        self.settings.setExternalCoreMLArtifactsDirectory(candidateDirectory, for: model)
        if self.settings.selectedSpeechModel == model {
            self.asr.resetTranscriptionProvider()
        } else {
            Task {
                await self.asr.checkIfModelsExistAsync()
            }
        }
        return true
    }
}
