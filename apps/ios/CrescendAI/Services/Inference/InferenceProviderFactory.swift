import Foundation

enum InferenceMode {
    case mock
    case coreML
}

enum InferenceProviderFactory {
    static var mode: InferenceMode {
        let useOnDevice = UserDefaults.standard.bool(forKey: "useOnDeviceInference")
        return useOnDevice ? .coreML : .mock
    }

    static func create() async -> any InferenceProvider {
        switch mode {
        case .mock:
            return MockInferenceProvider()
        case .coreML:
            let provider = CoreMLInferenceProvider()
            do {
                try await provider.loadModel()
            } catch {
                // Return the unloaded provider -- each infer() call will throw modelNotLoaded.
                // No silent fallback to mock.
            }
            return provider
        }
    }
}
