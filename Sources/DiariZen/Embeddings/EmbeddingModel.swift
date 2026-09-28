import CoreML
import Foundation

/// Speaker-embedding extractor. The kaldi fbank frontend is native Swift
/// (`KaldiFbank`); only the learned ResNet34 + masked pooling is CoreML. Maps an
/// active speech region (+ its per-frame speaker mask) to a fixed embedding vector.
public protocol EmbeddingModel {
    /// `region`: mono 16 kHz samples of the embedding window (16 s crop).
    /// `weights`: per-frame activity at segmentation resolution (799), or nil = all active.
    func embed(region: [Float], weights: [Float]?) throws -> [Float]
}

/// CoreML-backed embeddings: Swift fbank → CoreML ResNet34. Feature names + shapes
/// mirror `conversion/export_embedding_coreml.py`.
public struct CoreMLEmbedding: EmbeddingModel {
    /// Fixed CoreML input geometry.
    public static let cropSamples = 256_000  // 16 s window
    public static let segFrames = 799        // weights resolution (segmentation frames)
    static let inputFbank = "fbank"
    static let inputWeights = "weights"
    static let outputName = "embedding"

    let backend: CoreMLBackend
    let fbank: KaldiFbank

    public init(backend: CoreMLBackend, fbank: KaldiFbank = KaldiFbank()) {
        self.backend = backend
        self.fbank = fbank
    }

    public func embed(region: [Float], weights: [Float]?) throws -> [Float] {
        // Pad/truncate to the fixed 16 s crop so the fbank frame count matches the model.
        var window = region
        if window.count != Self.cropSamples {
            window = Array(window.prefix(Self.cropSamples))
            if window.count < Self.cropSamples {
                window.append(contentsOf: repeatElement(0, count: Self.cropSamples - window.count))
            }
        }

        let mels = fbank.compute(window)          // [F][80]
        let numFrames = mels.count
        let numMel = mels.first?.count ?? fbank.numMelBins

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
        let weightsArray = try MLMultiArray(
            shape: [1, NSNumber(value: w.count)], dataType: .float32)
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
