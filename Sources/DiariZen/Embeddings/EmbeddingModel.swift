import CoreML
import Foundation

/// Speaker-embedding extractor. The kaldi fbank frontend is native Swift
/// (`KaldiFbank`); only the learned ResNet34 + masked pooling is CoreML.
///
/// The fbank of a 16 s chunk is the SAME for every speaker in that chunk (only the
/// mask differs), so it is split out: compute it once per chunk with `fbank(region:)`,
/// then call `embed(fbank:weights:)` per speaker. `embed(region:weights:)` is a
/// convenience that does both.
public protocol EmbeddingModel {
    /// Kaldi log-mel features for a chunk (mono 16 kHz, padded to the 16 s window).
    func fbank(region: [Float]) -> [[Float]]
    /// 256-d embedding for one speaker: chunk `fbank` + per-frame `weights` mask
    /// (segmentation resolution, or nil = all active).
    func embed(fbank: [[Float]], weights: [Float]?) throws -> [Float]
}

public extension EmbeddingModel {
    /// Convenience: compute the fbank and embed one region in one call.
    func embed(region: [Float], weights: [Float]?) throws -> [Float] {
        try embed(fbank: fbank(region: region), weights: weights)
    }
}

/// CoreML-backed embeddings: Swift fbank → CoreML ResNet34. Feature names + shapes
/// mirror `conversion/export_embedding_coreml.py`.
public struct CoreMLEmbedding: EmbeddingModel {
    public static let cropSamples = 256_000  // 16 s window
    public static let segFrames = 799        // weights resolution (segmentation frames)
    static let inputFbank = "fbank"
    static let inputWeights = "weights"
    static let outputName = "embedding"

    let backend: CoreMLBackend
    let fbankFrontend: KaldiFbank

    public init(backend: CoreMLBackend, fbank: KaldiFbank = KaldiFbank()) {
        self.backend = backend
        self.fbankFrontend = fbank
    }

    public func fbank(region: [Float]) -> [[Float]] {
        // Pad/truncate to the fixed 16 s crop so the frame count matches the model.
        var window = region
        if window.count != Self.cropSamples {
            window = Array(window.prefix(Self.cropSamples))
            if window.count < Self.cropSamples {
                window.append(contentsOf: repeatElement(0, count: Self.cropSamples - window.count))
            }
        }
        return fbankFrontend.compute(window)
    }

    public func embed(fbank mels: [[Float]], weights: [Float]?) throws -> [Float] {
        let numFrames = mels.count
        let numMel = mels.first?.count ?? fbankFrontend.numMelBins

        let fbankArray = try MLMultiArray(
            shape: [1, NSNumber(value: numFrames), NSNumber(value: numMel)], dataType: .float32)
        fbankArray.withUnsafeMutableBytes { raw, _ in
            let dst = raw.bindMemory(to: Float32.self)
            var idx = 0
            for f in 0..<numFrames {
                let row = mels[f]
                for m in 0..<numMel { dst[idx] = row[m]; idx += 1 }
            }
        }

        let w = weights ?? [Float](repeating: 1, count: Self.segFrames)
        let weightsArray = try MLMultiArray(shape: [1, NSNumber(value: w.count)], dataType: .float32)
        weightsArray.withUnsafeMutableBytes { raw, _ in
            let dst = raw.bindMemory(to: Float32.self)
            for i in 0..<w.count { dst[i] = w[i] }
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            Self.inputFbank: MLFeatureValue(multiArray: fbankArray),
            Self.inputWeights: MLFeatureValue(multiArray: weightsArray),
        ])
        let out: MLFeatureProvider
        do {
            out = try backend.model.prediction(from: provider)
        } catch {
            throw DiariZenError.inference("embedding predict: \(error)")
        }
        guard let arr = out.featureValue(for: Self.outputName)?.multiArrayValue else {
            throw DiariZenError.inference("embedding output '\(Self.outputName)' missing")
        }
        return (0..<arr.count).map { arr[$0].floatValue }
    }
}
