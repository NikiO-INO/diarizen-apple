import Foundation

/// Speaker-embedding extractor (may be a separate CoreML model, or fused into
/// segmentation — confirmed in Phase 0). Maps an active speech region to a fixed
/// embedding vector used by clustering.
public protocol EmbeddingModel {
    func embed(region: [Float]) throws -> [Float]
}

/// CoreML-backed embeddings. TODO(phase2): feature names + output length.
public struct CoreMLEmbedding: EmbeddingModel {
    let backend: CoreMLBackend
    public init(backend: CoreMLBackend) { self.backend = backend }

    public func embed(region: [Float]) throws -> [Float] {
        throw DiariZenError.notImplemented("CoreMLEmbedding.embed (phase 2)")
    }
}
