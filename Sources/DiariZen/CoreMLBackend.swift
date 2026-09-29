import CoreML
import Foundation

/// Thin wrapper around a compiled CoreML model (`.mlmodelc`) with configurable
/// compute units. All neural stages load through this so we control the ANE/GPU/
/// CPU policy in one place.
public struct CoreMLBackend {

    /// Which compute units CoreML may use.
    ///
    /// Default is `.cpuAndGPU`, the fastest measured policy for this model
    /// (RTF 0.055; see benchmarks/RESULTS.md). Findings on the WavLM+Conformer
    /// segmentation model (Apple Silicon):
    ///   - `.all` (CPU+GPU+ANE) **DEADLOCKS** — the first predict hangs indefinitely
    ///     at ~0% CPU. This is specifically the GPU+ANE partitioning; avoid it.
    ///   - `.cpuAndNeuralEngine` runs, but WavLM segmentation is ~12× SLOWER on the
    ///     ANE than on the GPU (0.40 s vs 0.033 s per chunk) and uses ~5× the memory,
    ///     so it is not worth using for this model. Kept as an opt-in.
    ///   - `.cpuAndGPU` / `.cpuOnly` are fine; GPU is fastest overall.
    public enum ComputePolicy: Sendable {
        case cpuAndGPU  // .cpuAndGPU — production default, fastest here
        case all  // .all — CPU + GPU + Neural Engine (DEADLOCKS — do not use)
        case cpuAndNeuralEngine  // .cpuAndNeuralEngine (works, but slow for WavLM seg)
        case cpuOnly  // .cpuOnly — parity/debug

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
