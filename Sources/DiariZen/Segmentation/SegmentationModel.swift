import Foundation

/// WavLM + Conformer powerset segmentation, one CoreML model. Consumes a fixed
/// audio chunk, produces per-frame speaker-activity (powerset) logits/probs.
public protocol SegmentationModel {
    /// Frames of per-speaker activity for one chunk. Shape/decoding are defined
    /// by the exported model (Phase 1/2). Returns opaque frames the pipeline
    /// decodes downstream.
    func segment(chunk: [Float]) throws -> SegmentationOutput
}

/// Raw segmentation output for one chunk. TODO(phase2): shape it to the real
/// model (e.g. [frames, powerset_classes] or [frames, maxSpeakers]).
public struct SegmentationOutput: Sendable {
    public let frames: [[Float]]
    public init(frames: [[Float]]) { self.frames = frames }
}

/// CoreML-backed implementation. TODO(phase2): wire input/output feature names
/// (from export_coreml.py) and the frame decoding.
public struct CoreMLSegmentation: SegmentationModel {
    let backend: CoreMLBackend
    public init(backend: CoreMLBackend) { self.backend = backend }

    public func segment(chunk: [Float]) throws -> SegmentationOutput {
        throw DiariZenError.notImplemented("CoreMLSegmentation.segment (phase 2)")
    }
}
