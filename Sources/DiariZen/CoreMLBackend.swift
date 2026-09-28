import CoreML
import Foundation

/// Thin wrapper around a compiled CoreML model (`.mlmodelc`) with configurable
/// compute units. All neural stages load through this so we control the ANE/GPU/
/// CPU policy in one place.
public struct CoreMLBackend {

    /// Which compute units CoreML may use.
    ///
    /// Default is `.cpuAndGPU`, NOT `.all`. The converted WavLM+Conformer
    /// segmentation model DEADLOCKS on the first Neural Engine predict on Apple
    /// Silicon — the call hangs indefinitely at ~0% CPU (reproduced in the Python
    /// parity harness with `compute_units=.all`; CPU/GPU predict returns in
    /// milliseconds). Until that ANE path is profiled and fixed (Phase 4), any
    /// policy that includes the ANE (`.all`, `.cpuAndNeuralEngine`) is opt-in and
    /// treated as experimental — do not ship it as the default.
    public enum ComputePolicy: Sendable {
        case cpuAndGPU              // .cpuAndGPU — safe production default (no ANE hang)
        case all                   // .all — CPU + GPU + Neural Engine (EXPERIMENTAL: ANE deadlock)
        case cpuAndNeuralEngine    // .cpuAndNeuralEngine (EXPERIMENTAL: ANE deadlock)
        case cpuOnly               // .cpuOnly — parity/debug

        var mlComputeUnits: MLComputeUnits {
            switch self {
            case .cpuAndGPU: return .cpuAndGPU
            case .all: return .all
            case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
            case .cpuOnly: return .cpuOnly
            }
        }
    }

    public let model: MLModel

    /// Load a CoreML model from either a compiled `.mlmodelc` or an `.mlpackage`
    /// (compiled on the fly). `MLModel(contentsOf:)` only accepts a compiled model,
    /// so an `.mlpackage` — what `export_coreml.py` produces — is compiled first.
    public init(modelURL: URL, policy: ComputePolicy = .cpuAndGPU) throws {
        let config = MLModelConfiguration()
        config.computeUnits = policy.mlComputeUnits

        let compiledURL: URL
        if modelURL.pathExtension == "mlmodelc" {
            compiledURL = modelURL
        } else {
            do {
                compiledURL = try MLModel.compileModel(at: modelURL)
            } catch {
                throw DiariZenError.modelLoad("compile \(modelURL.lastPathComponent): \(error)")
            }
        }

        do {
            self.model = try MLModel(contentsOf: compiledURL, configuration: config)
        } catch {
            throw DiariZenError.modelLoad("\(compiledURL.lastPathComponent): \(error)")
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
