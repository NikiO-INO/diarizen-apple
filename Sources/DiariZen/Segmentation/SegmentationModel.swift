import CoreML
import Foundation

/// WavLM + Conformer powerset segmentation, one CoreML model. Consumes a fixed
/// audio chunk, produces per-frame speaker-activity (powerset) log-probs.
public protocol SegmentationModel {
    /// Powerset log-probs for one fixed chunk: `[numFrames][numClasses]`.
    /// The pipeline decodes the powerset frames downstream (Phase 3).
    func segment(chunk: [Float]) throws -> SegmentationOutput
}

/// Raw segmentation output for one chunk: `frames[f][c]` is the log-prob of
/// powerset class `c` at frame `f` (799 frames × 11 classes for base-s80-md).
public struct SegmentationOutput: Sendable {
    public let frames: [[Float]]
    public init(frames: [[Float]]) { self.frames = frames }
}

/// CoreML-backed segmentation. Input/output feature names and the static input
/// geometry mirror `conversion/export_coreml.py`.
public struct CoreMLSegmentation: SegmentationModel {
    /// Fixed model input geometry: 16 s × 16 kHz mono = 256000 samples.
    public static let windowSamples = 256_000
    static let inputName = "waveform"
    static let outputName = "segmentation"

    let backend: CoreMLBackend
    public init(backend: CoreMLBackend) { self.backend = backend }

    public func segment(chunk: [Float]) throws -> SegmentationOutput {
        let n = Self.windowSamples
        // Static (1, 1, 256000) input — pad or truncate a partial chunk.
        let input = try MLMultiArray(shape: [1, 1, NSNumber(value: n)], dataType: .float32)
        input.withUnsafeMutableBytes { raw, _ in
            let dst = raw.bindMemory(to: Float32.self)
            let count = min(chunk.count, n)
            for i in 0..<count { dst[i] = chunk[i] }
            for i in count..<n { dst[i] = 0 }
        }

        let out = try backend.predict(
            featureName: Self.inputName,
            value: MLFeatureValue(multiArray: input)
        )
        guard let arr = out.featureValue(for: Self.outputName)?.multiArrayValue else {
            throw DiariZenError.inference("segmentation output '\(Self.outputName)' missing")
        }
        return SegmentationOutput(frames: Self.toFrames(arr))
    }

    /// (1, frames, classes) MLMultiArray → `[frames][classes]` Float. C-contiguous,
    /// so element (0, f, c) lives at linear index `f*classes + c`. dtype-agnostic.
    static func toFrames(_ arr: MLMultiArray) -> [[Float]] {
        let frames = arr.shape[1].intValue
        let classes = arr.shape[2].intValue
        var result = [[Float]]()
        result.reserveCapacity(frames)

        if arr.dataType == .float32 {
            arr.withUnsafeBytes { raw in
                let src = raw.bindMemory(to: Float32.self)
                let base = src.baseAddress!
                for f in 0..<frames {
                    result.append(Array(UnsafeBufferPointer(start: base + f * classes, count: classes)))
                }
            }
        } else {
            for f in 0..<frames {
                var row = [Float](repeating: 0, count: classes)
                for c in 0..<classes {
                    row[c] = arr[[0, NSNumber(value: f), NSNumber(value: c)]].floatValue
                }
                result.append(row)
            }
        }
        return result
    }
}
