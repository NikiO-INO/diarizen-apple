import Foundation

/// Segmentation frame resolution (pyannote receptive field): each per-chunk frame
/// spans `duration` s and advances `step` s. Fixed for a given segmentation model.
public struct FrameResolution: Sendable {
    public let start: Double, duration: Double, step: Double
    public init(start: Double, duration: Double, step: Double) {
        self.start = start; self.duration = duration; self.step = step
    }
}

/// Build final speaker turns from per-chunk binarized segmentation + cluster labels.
/// Ports pyannote `reconstruct` → `to_diarization` (overlap-add `aggregate` +
/// per-frame top-`count` selection) → `Binarize`.
public enum Reconstruction {
    public static func turns(
        binarized: [[[Float]]],   // [chunk][frame][localSpeaker], 0/1
        hardClusters: [[Int]],    // [chunk][localSpeaker], -2 = unassigned
        window: SlidingWindow,    // chunk window (start 0, duration 16, step 1.6)
        frame: FrameResolution,
        sampleRate: Int
    ) -> [SpeakerTurn] {
        let numChunks = binarized.count
        guard numChunks > 0 else { return [] }
        let framesPerChunk = binarized[0].count
        let localSpeakers = binarized[0][0].count

        let maxCluster = hardClusters.flatMap { $0 }.max() ?? -1
        let numClusters = maxCluster + 1
        guard numClusters > 0 else { return [] }

        // reconstruct: clustered[c][f][k] = max over local speakers assigned to k.
        var clustered = zeros3(numChunks, framesPerChunk, numClusters)
        var clusteredMask = mask3(numChunks, framesPerChunk, numClusters)
        for c in 0..<numChunks {
            for k in Set(hardClusters[c].filter { $0 != -2 }) {
                let members = (0..<localSpeakers).filter { hardClusters[c][$0] == k }
                for f in 0..<framesPerChunk {
                    var mx = 0.0
                    for s in members { mx = Swift.max(mx, Double(binarized[c][f][s])) }
                    clustered[c][f][k] = mx
                    clusteredMask[c][f][k] = true
                }
            }
        }

        // speaker_count: aggregate(Σ_speakers binarized, averaged) then round.
        var sumScores = zeros3(numChunks, framesPerChunk, 1)
        let sumMask = mask3(numChunks, framesPerChunk, 1, value: true)
        for c in 0..<numChunks { for f in 0..<framesPerChunk {
            var s = 0.0; for spk in 0..<localSpeakers { s += Double(binarized[c][f][spk]) }
            sumScores[c][f][0] = s
        } }
        let countAgg = aggregate(sumScores, sumMask, window, numChunks, frame, skipAverage: false, missing: 0)
        let count = countAgg.map { Int($0[0].rounded(.toNearestOrEven)) }

        // to_diarization: aggregate clustered (raw sum), select top-count per frame.
        let activations = aggregate(clustered, clusteredMask, window, numChunks, frame, skipAverage: true, missing: 0)
        let T = Swift.min(activations.count, count.count)
        let maxCount = count.prefix(T).max() ?? 0
        let numOut = Swift.max(numClusters, maxCount)

        var binary = zeros2(T, numOut)
        for t in 0..<T {
            let c = count[t]
            guard c > 0 else { continue }
            // pad missing columns with 0 activation, argsort descending.
            let acts = (0..<numOut).map { $0 < numClusters ? activations[t][$0] : 0.0 }
            let order = (0..<numOut).sorted { acts[$0] > acts[$1] }
            for i in 0..<Swift.min(c, numOut) { binary[t][order[i]] = 1 }
        }

        return binarize(binary, frame: frame)
    }

    /// pyannote `Inference.aggregate` (hamming=False, warm_up=(0,0)) overlap-add.
    static func aggregate(
        _ scores: [[[Double]]], _ masks: [[[Bool]]],
        _ window: SlidingWindow, _ numChunks: Int, _ frame: FrameResolution,
        skipAverage: Bool, missing: Double
    ) -> [[Double]] {
        let framesPerChunk = scores[0].count
        let numClasses = scores[0][0].count
        let fStart = window.start
        func closestFrame(_ t: Double) -> Int {
            Int(((t - fStart - 0.5 * frame.duration) / frame.step).rounded(.toNearestOrEven))
        }
        let lastEnd = window.start + window.duration + Double(numChunks - 1) * window.step + 0.5 * frame.duration
        let numFrames = closestFrame(lastEnd) + 1
        guard numFrames > 0 else { return [] }

        var agg = zeros2(numFrames, numClasses)
        var cnt = zeros2(numFrames, numClasses)
        var aggMask = zeros2(numFrames, numClasses)
        for c in 0..<numChunks {
            let chunkStart = window.start + Double(c) * window.step
            let startFrame = closestFrame(chunkStart + 0.5 * frame.duration)
            for fp in 0..<framesPerChunk {
                let gf = startFrame + fp
                if gf < 0 || gf >= numFrames { continue }
                for k in 0..<numClasses {
                    let m = masks[c][fp][k] ? 1.0 : 0.0
                    agg[gf][k] += scores[c][fp][k] * m
                    cnt[gf][k] += m
                    if m > aggMask[gf][k] { aggMask[gf][k] = m }
                }
            }
        }
        var out = agg
        if !skipAverage {
            for i in 0..<numFrames { for k in 0..<numClasses { out[i][k] = agg[i][k] / Swift.max(cnt[i][k], 1e-12) } }
        }
        for i in 0..<numFrames { for k in 0..<numClasses where aggMask[i][k] == 0 { out[i][k] = missing } }
        return out
    }

    /// pyannote `Binarize` (onset=offset=0.5, no padding/min-duration) → turns.
    static func binarize(_ scores: [[Double]], frame: FrameResolution) -> [SpeakerTurn] {
        let numFrames = scores.count
        guard numFrames > 0 else { return [] }
        let numClasses = scores[0].count
        func ts(_ i: Int) -> Double { frame.start + Double(i) * frame.step + 0.5 * frame.duration }
        let onset = 0.5, offset = 0.5

        var turns = [SpeakerTurn]()
        for k in 0..<numClasses {
            var start = ts(0)
            var isActive = scores[0][k] > onset
            var lastT = ts(0)
            for i in 1..<numFrames {
                let t = ts(i); lastT = t
                let y = scores[i][k]
                if isActive {
                    if y < offset { turns.append(SpeakerTurn(start: start, end: t, speaker: k)); start = t; isActive = false }
                } else if y > onset {
                    start = t; isActive = true
                }
            }
            if isActive { turns.append(SpeakerTurn(start: start, end: lastT, speaker: k)) }
        }
        return turns
    }
}

private func zeros2(_ a: Int, _ b: Int) -> [[Double]] {
    [[Double]](repeating: [Double](repeating: 0, count: b), count: a)
}
private func zeros3(_ a: Int, _ b: Int, _ c: Int) -> [[[Double]]] {
    [[[Double]]](repeating: zeros2(b, c), count: a)
}
private func mask3(_ a: Int, _ b: Int, _ c: Int, value: Bool = false) -> [[[Bool]]] {
    [[[Bool]]](repeating: [[Bool]](repeating: [Bool](repeating: value, count: c), count: b), count: a)
}
