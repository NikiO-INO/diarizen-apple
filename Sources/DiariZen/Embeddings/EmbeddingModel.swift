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
    /// Batched embed: `fbanks[i]` + `weights[i]` → `result[i]`. One CoreML call for
    /// the whole batch (amortizes per-call overhead; embedding dominates runtime).
    func embed(fbanks: [[[Float]]], weights: [[Float]]) throws -> [[Float]]
}

public extension EmbeddingModel {
    /// Convenience: compute the fbank and embed one region in one call.
    func embed(region: [Float], weights: [Float]?) throws -> [Float] {
        try embed(fbank: fbank(region: region), weights: weights)
    }
    /// Fallback batched embed (per-item loop) for non-batching implementations.
    func embed(fbanks: [[[Float]]], weights: [[Float]]) throws -> [[Float]] {
        try zip(fbanks, weights).map { try embed(fbank: $0.0, weights: $0.1) }
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

    public func embed(fbanks: [[[Float]]], weights: [[Float]]) throws -> [[Float]] {
        let b = fbanks.count
        guard b > 0 else { return [] }
        let numFrames = fbanks[0].count
        let numMel = fbanks[0].first?.count ?? fbankFrontend.numMelBins
        let wLen = weights.first?.count ?? Self.segFrames

        let fbankArray = try MLMultiArray(
            shape: [NSNumber(value: b), NSNumber(value: numFrames), NSNumber(value: numMel)],
            dataType: .float32)
        fbankArray.withUnsafeMutableBytes { raw, _ in
            let dst = raw.bindMemory(to: Float32.self)
            var idx = 0
            for i in 0..<b {
                let mels = fbanks[i]
                for f in 0..<numFrames { let row = mels[f]; for m in 0..<numMel { dst[idx] = row[m]; idx += 1 } }
            }
        }
        let weightsArray = try MLMultiArray(
            shape: [NSNumber(value: b), NSNumber(value: wLen)], dataType: .float32)
        weightsArray.withUnsafeMutableBytes { raw, _ in
            let dst = raw.bindMemory(to: Float32.self)
            var idx = 0
            for i in 0..<b { let wr = weights[i]; for k in 0..<wLen { dst[idx] = wr[k]; idx += 1 } }
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            Self.inputFbank: MLFeatureValue(multiArray: fbankArray),
            Self.inputWeights: MLFeatureValue(multiArray: weightsArray),
        ])
        let out: MLFeatureProvider
        do {
            out = try backend.model.prediction(from: provider)
        } catch {
            throw DiariZenError.inference("embedding batch predict: \(error)")
        }
        guard let arr = out.featureValue(for: Self.outputName)?.multiArrayValue else {
            throw DiariZenError.inference("embedding output '\(Self.outputName)' missing")
        }
        let dim = arr.shape[1].intValue
        var result = [[Float]](repeating: [Float](repeating: 0, count: dim), count: b)
        if arr.dataType == .float32 {
            arr.withUnsafeBytes { raw in
                let src = raw.bindMemory(to: Float32.self)
                for i in 0..<b { let base = i * dim; for k in 0..<dim { result[i][k] = src[base + k] } }
            }
        } else {
            for i in 0..<b { let base = i * dim; for k in 0..<dim { result[i][k] = arr[base + k].floatValue } }
        }
        return result
    }
}
