import Foundation

/// A single diarized turn on the audio timeline.
public struct SpeakerTurn: Equatable, Sendable {
    public let start: Double   // seconds
    public let end: Double     // seconds
    public let speaker: Int    // 1-based, stable within a run
    public init(start: Double, end: Double, speaker: Int) {
        self.start = start
        self.end = end
        self.speaker = speaker
    }
}

/// End-to-end DiariZen pipeline on Apple Silicon. Deliberately modular: each
/// neural stage is its own CoreML model, and aggregation/clustering are plain
/// Swift — never one monolithic graph.
///
///   audio → segmentation (CoreML) → aggregation (Swift)
///         → embeddings (CoreML) → clustering/VBx (Swift) → turns
public final class DiarizationPipeline {
    private let segmentation: SegmentationModel
    private let embeddings: EmbeddingModel
    private let clustering: Clustering

    public init(
        segmentation: SegmentationModel,
        embeddings: EmbeddingModel,
        clustering: Clustering = VBxClustering()
    ) {
        self.segmentation = segmentation
        self.embeddings = embeddings
        self.clustering = clustering
    }

    /// Diarize a mono 16 kHz waveform into speaker turns.
    /// TODO(phase2/3): implement chunk sliding, powerset decoding, embedding
    /// extraction over active regions, and clustering. This signature is the
    /// stable contract the CLI + host integrations depend on.
    public func diarize(waveform: [Float], sampleRate: Int) throws -> [SpeakerTurn] {
        _ = try segmentation.segment(chunk: waveform)   // placeholder wiring
        throw DiariZenError.notImplemented("DiarizationPipeline.diarize (phases 2–3)")
    }
}

public enum DiariZenError: Error, CustomStringConvertible {
    case notImplemented(String)
    case modelLoad(String)
    case inference(String)

    public var description: String {
        switch self {
        case .notImplemented(let s): return "not implemented: \(s)"
        case .modelLoad(let s): return "model load failed: \(s)"
        case .inference(let s): return "inference failed: \(s)"
        }
    }
}
