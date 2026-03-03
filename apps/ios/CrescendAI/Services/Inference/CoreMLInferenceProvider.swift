import CoreML
import Foundation
import Synchronization

enum CoreMLInferenceError: Error {
    case modelNotFound
    case modelLoadFailed(underlying: Error)
    case predictionFailed(underlying: Error)
    case unexpectedOutputFormat
    case modelNotLoaded
}

final class CoreMLInferenceProvider: @unchecked Sendable {
    private static let dimensionKeys = [
        "dynamics", "timing", "pedaling",
        "articulation", "phrasing", "interpretation"
    ]

    private let lock = NSLock()
    private var _model: MLModel?

    var isModelLoaded: Bool {
        lock.withLock { _model != nil }
    }

    func loadModel() async throws {
        guard let modelURL = Bundle.main.url(forResource: "CrescendMuQ", withExtension: "mlmodelc") else {
            throw CoreMLInferenceError.modelNotFound
        }

        let config = MLModelConfiguration()
        config.computeUnits = .all

        let loadedModel: MLModel
        do {
            loadedModel = try await MLModel.load(contentsOf: modelURL, configuration: config)
        } catch {
            throw CoreMLInferenceError.modelLoadFailed(underlying: error)
        }

        storeModel(loadedModel)
    }

    private func storeModel(_ model: MLModel) {
        lock.withLock { _model = model }
    }

    private func getModel() -> MLModel? {
        lock.withLock { _model }
    }
}

// MARK: - InferenceProvider

extension CoreMLInferenceProvider: InferenceProvider {
    func infer(samples: [Float], sampleRate: Int) async throws -> InferenceResult {
        guard let model = getModel() else {
            throw CoreMLInferenceError.modelNotLoaded
        }

        let startTime = CFAbsoluteTimeGetCurrent()

        // Create MLMultiArray input: shape [1, numSamples]
        let shape: [NSNumber] = [1, NSNumber(value: samples.count)]
        let input: MLMultiArray
        do {
            input = try MLMultiArray(shape: shape, dataType: .float32)
        } catch {
            throw CoreMLInferenceError.predictionFailed(underlying: error)
        }

        let pointer = input.dataPointer.bindMemory(to: Float.self, capacity: samples.count)
        for i in 0..<samples.count {
            pointer[i] = samples[i]
        }

        let featureProvider: MLDictionaryFeatureProvider
        do {
            featureProvider = try MLDictionaryFeatureProvider(
                dictionary: ["audio": MLFeatureValue(multiArray: input)]
            )
        } catch {
            throw CoreMLInferenceError.predictionFailed(underlying: error)
        }

        let prediction: MLFeatureProvider
        do {
            prediction = try await model.prediction(from: featureProvider)
        } catch {
            throw CoreMLInferenceError.predictionFailed(underlying: error)
        }

        let elapsedMs = Int((CFAbsoluteTimeGetCurrent() - startTime) * 1000)
        let dimensions = try parsePrediction(prediction)

        return InferenceResult(dimensions: dimensions, processingTimeMs: elapsedMs)
    }

    private func parsePrediction(_ prediction: MLFeatureProvider) throws -> [String: Float] {
        // Strategy 1: Single "scores" output array of shape [6]
        if let scoresFeature = prediction.featureValue(for: "scores"),
           let scoresArray = scoresFeature.multiArrayValue {
            guard scoresArray.count == Self.dimensionKeys.count else {
                throw CoreMLInferenceError.unexpectedOutputFormat
            }
            var dimensions: [String: Float] = [:]
            let scoresPointer = scoresArray.dataPointer.bindMemory(
                to: Float.self, capacity: Self.dimensionKeys.count
            )
            for (i, key) in Self.dimensionKeys.enumerated() {
                dimensions[key] = scoresPointer[i]
            }
            return dimensions
        }

        // Strategy 2: Six named outputs (one per dimension)
        var dimensions: [String: Float] = [:]
        for key in Self.dimensionKeys {
            guard let feature = prediction.featureValue(for: key) else {
                throw CoreMLInferenceError.unexpectedOutputFormat
            }
            if let array = feature.multiArrayValue {
                let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: 1)
                dimensions[key] = ptr[0]
            } else if feature.type == .double {
                dimensions[key] = Float(feature.doubleValue)
            } else {
                throw CoreMLInferenceError.unexpectedOutputFormat
            }
        }
        return dimensions
    }
}
