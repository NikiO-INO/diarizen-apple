import Foundation

/// Speaker clustering / post-processing — CPU-only Swift, never a CoreML graph.
/// DiariZen uses VBx (Bayesian HMM over x-vectors) on top of an AHC warm start.
public protocol Clustering {
    /// Assign each embedding to a speaker id. `embeddings[i]` corresponds to the
    /// active region at `times[i]` (start, end) seconds.
    func cluster(embeddings: [[Float]], times: [(Double, Double)]) throws -> [SpeakerTurn]
}

/// VBx clustering. TODO(phase3): port the VBx/AHC + powerset overlap handling and
/// validate against DiariZen's Python clustering on identical segmentation
/// outputs. May reuse an existing VBx implementation.
public struct VBxClustering: Clustering {
    /// AHC cut / VBx hyper-parameters — keep the same values across datasets, as
    /// DiariZen does. Real defaults set in Phase 3 from the upstream config.
    public var ahcThreshold: Double
    public var fa: Double
    public var fb: Double

    public init(ahcThreshold: Double = 0.6, fa: Double = 0.07, fb: Double = 0.8) {
        self.ahcThreshold = ahcThreshold
        self.fa = fa
        self.fb = fb
    }

    public func cluster(embeddings: [[Float]], times: [(Double, Double)]) throws -> [SpeakerTurn] {
        throw DiariZenError.notImplemented("VBxClustering.cluster (phase 3)")
    }
}
