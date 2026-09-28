import CoreML
import Foundation

/// Thin wrapper around a compiled CoreML model (`.mlmodelc`) with configurable
/// compute units. All neural stages load through this so we control the ANE/GPU/
/// CPU policy in one place.
public struct CoreMLBackend {

    /// Which compute units CoreML may use. Default keeps the ANE path available;
    /// Phase 4 profiling decides the shipped default.
    public enum ComputePolicy: Sendable {
        case all                    // .all  — CPU + GPU + Neural Engine
        case cpuAndNeuralEngine     // .cpuAndNeuralEngine
        case cpuOnly                // .cpuOnly — parity/debug

        var mlComputeUnits: MLComputeUnits {
            switch self {
            case .all: return .all
            case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
            case .cpuOnly: return .cpuOnly
            }
        }
    }

    public let model: MLModel

    public init(compiledModelURL: URL, policy: ComputePolicy = .all) throws {
        let config = MLModelConfiguration()
        config.computeUnits = policy.mlComputeUnits
        do {
            self.model = try MLModel(contentsOf: compiledModelURL, configuration: config)
        } catch {
            throw DiariZenError.modelLoad("\(compiledModelURL.lastPathComponent): \(error)")
        }
    }

    /// Run a single-input, single-output float model. Real stage wrappers
    /// (Segmentation/Embeddings) build on this with their own feature names and
    /// shapes.
    func predict(featureName: String, value: MLFeatureValue) throws -> MLFeatureProvider {
        let provider = try MLDictionaryFeatureProvider(dictionary: [featureName: value])
        do {
            return try model.prediction(from: provider)
        } catch {
            throw DiariZenError.inference("\(error)")
        }
    }
}
