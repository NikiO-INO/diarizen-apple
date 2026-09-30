import CoreML
import Foundation

/// Thin wrapper around a compiled CoreML model (`.mlmodelc`) with configurable
/// compute units. All neural stages load through this so we control the ANE/GPU/
/// CPU policy in one place.
public struct CoreMLBackend {

    /// Which compute units CoreML may use.
    ///
    /// Default is `.cpuAndGPU`, the fastest measured policy for this model
    /// (RTF 0.037; see benchmarks/RESULTS.md). Findings on the WavLM+Conformer
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

    /// Load a CoreML model from either a compiled `.mlmodelc` or an `.mlpackage`.
    /// `MLModel(contentsOf:)` only accepts a compiled model, so an `.mlpackage` —
    /// what `export_coreml.py` produces — is compiled first, and the compiled
    /// result is cached as a sibling `.mlmodelc` so repeated runs skip the compile.
    public init(modelURL: URL, policy: ComputePolicy = .cpuAndGPU) throws {
        let config = MLModelConfiguration()
        config.computeUnits = policy.mlComputeUnits

        let compiledURL: URL
        if modelURL.pathExtension == "mlmodelc" {
            compiledURL = modelURL
        } else {
            compiledURL = try Self.compiledModel(for: modelURL)
        }

        do {
            self.model = try MLModel(contentsOf: compiledURL, configuration: config)
        } catch {
            throw DiariZenError.modelLoad("\(compiledURL.lastPathComponent): \(error)")
        }
    }

    /// Return a compiled `.mlmodelc` for an `.mlpackage`, reusing a cached sibling
    /// `.mlmodelc` when it is present and up to date. Compiling an `.mlpackage` is
    /// slow, and a host that spawns the CLI once per file would otherwise pay that
    /// cost on every run; caching turns it into a one-time cost. Falls back to a
    /// fresh temporary compile if the cache cannot be written (e.g. a read-only dir).
    static func compiledModel(for packageURL: URL) throws -> URL {
        let cacheURL = packageURL.deletingPathExtension().appendingPathExtension("mlmodelc")
        if cacheIsFresh(cacheURL: cacheURL, sourceURL: packageURL) {
            return cacheURL
        }
        let compiled: URL
        do {
            compiled = try MLModel.compileModel(at: packageURL)
        } catch {
            throw DiariZenError.modelLoad("compile \(packageURL.lastPathComponent): \(error)")
        }
        let fm = FileManager.default
        do {
            if fm.fileExists(atPath: cacheURL.path) { try fm.removeItem(at: cacheURL) }
            try fm.copyItem(at: compiled, to: cacheURL)
            return cacheURL
        } catch {
            return compiled  // cache write failed (e.g. read-only dir); use the temp compile
        }
    }

    /// A cached `.mlmodelc` is usable when it exists and is at least as new as its
    /// source `.mlpackage`, so re-exporting the model invalidates the cache.
    static func cacheIsFresh(cacheURL: URL, sourceURL: URL) -> Bool {
        let fm = FileManager.default
        guard fm.fileExists(atPath: cacheURL.path) else { return false }
        let modDate: (URL) -> Date? = { url in
            (try? fm.attributesOfItem(atPath: url.path))?[.modificationDate] as? Date
        }
        guard let cacheMod = modDate(cacheURL), let srcMod = modDate(sourceURL) else { return false }
        return cacheMod >= srcMod
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
