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

/// Chunk-time geometry of the segmentation sliding window (pyannote `SlidingWindow`).
public struct SlidingWindow: Sendable {
    public let start: Double     // 0.0 s
    public let duration: Double  // 16.0 s (segmentation window)
    public let step: Double      // 1.6 s
    public init(start: Double, duration: Double, step: Double) {
        self.start = start; self.duration = duration; self.step = step
    }
}

/// Everything Phase-3 clustering consumes: per-chunk binarized speaker activity and
/// per-(chunk, speaker) embeddings, plus the window geometry to place chunks in time.
public struct DiarizationFeatures: Sendable {
    public let binarized: [[[Float]]]    // [chunk][frame][speaker], 0/1
    public let embeddings: [[[Float]]]   // [chunk][speaker][dim]
    public let window: SlidingWindow
    public init(binarized: [[[Float]]], embeddings: [[[Float]]], window: SlidingWindow) {
        self.binarized = binarized; self.embeddings = embeddings; self.window = window
    }
}

/// End-to-end DiariZen pipeline on Apple Silicon. Deliberately modular: each
/// neural stage is its own CoreML model, and aggregation/clustering are plain
/// Swift — never one monolithic graph.
///
///   audio → segmentation (CoreML) → aggregation (Swift)
///         → embeddings (CoreML) → clustering/VBx (Swift) → turns
/// Per-stage wall-clock (seconds) for one `diarize` run — for benchmarking.
public struct StageTimings: Sendable {
    public var segmentation = 0.0
    public var embedding = 0.0
    public var clustering = 0.0
    public var reconstruction = 0.0
    public var total = 0.0
}

public final class DiarizationPipeline {
    private let segmentation: SegmentationModel
    private let embeddings: EmbeddingModel
    private let clustering: VBxClustering?
    private let powerset: Powerset

    /// Wall-clock breakdown of the most recent `diarize` call.
    public private(set) var timings = StageTimings()

    // Pipeline constants (base-s80-md), mirroring the DiariZen config + pyannote.
    static let windowSamples = 256_000  // 16 s @ 16 kHz
    static let stepSeconds = 1.6        // segmentation_step * seg_duration = 0.1 * 16
    static let medianWindow = 11        // median_filter size (1, 11, 1)
    static let minNumFrames = 2         // ceil(799 * min_num_samples/num_samples) for embeddings
    // Segmentation receptive field (base-s80-md): 799 frames of 25 ms every 20 ms.
    static let frameResolution = FrameResolution(start: -0.00753125, duration: 0.025, step: 0.02)

    public init(
        segmentation: SegmentationModel,
        embeddings: EmbeddingModel,
        clustering: VBxClustering? = nil,
        powerset: Powerset = Powerset(numSpeakers: 4, maxSetSize: 2)
    ) {
        self.segmentation = segmentation
        self.embeddings = embeddings
        self.clustering = clustering
        self.powerset = powerset
    }

    /// Diarize a mono 16 kHz waveform into speaker turns: Phase-2 features →
    /// VBx clustering → reconstruction → binarized turns.
    public func diarize(waveform: [Float], sampleRate: Int) throws -> [SpeakerTurn] {
        guard let clustering else {
            throw DiariZenError.modelLoad("clustering (PLDA) not configured")
        }
        timings = StageTimings()
        let t0 = Date()
        let feats = try extractFeatures(waveform: waveform, sampleRate: sampleRate)

        let tCluster = Date()
        var hard = clustering.cluster(embeddings: feats.embeddings, binarized: feats.binarized)
        // Inactive speakers (no active frame in a chunk) → -2, as in the pipeline.
        for c in feats.binarized.indices {
            for s in 0..<powerset.numSpeakers {
                if !feats.binarized[c].contains(where: { $0[s] > 0.5 }) { hard[c][s] = -2 }
            }
        }
        let tRecon = Date()
        timings.clustering = tRecon.timeIntervalSince(tCluster)
        let turns = Reconstruction.turns(
            binarized: feats.binarized, hardClusters: hard, window: feats.window,
            frame: Self.frameResolution, sampleRate: sampleRate
        )
        timings.reconstruction = Date().timeIntervalSince(tRecon)
        timings.total = Date().timeIntervalSince(t0)
        return turns
    }

    /// Phase 2: segmentation sliding window → powerset decode → median filter →
    /// per-(chunk, speaker) embeddings. Reproduces pyannote's get_segmentations +
    /// get_embeddings (exclude_overlap) up to clustering.
    public func extractFeatures(waveform: [Float], sampleRate: Int) throws -> DiarizationFeatures {
        let windowSamples = Self.windowSamples
        let stepSamples = Int((Self.stepSeconds * Double(sampleRate)).rounded())
        let n = waveform.count

        // Chunk starts: complete windows via unfold, plus a zero-padded last chunk.
        var starts = [Int]()
        if n >= windowSamples {
            var c = 0
            while c * stepSamples + windowSamples <= n { starts.append(c * stepSamples); c += 1 }
        }
        let numComplete = starts.count
        let hasLast = (n < windowSamples) || ((n - windowSamples) % stepSamples > 0)
        if hasLast { starts.append(numComplete * stepSamples) }

        let numSpeakers = powerset.numSpeakers
        var binarized = [[[Float]]]()      // [chunk][frame][speaker]
        var windows = [[Float]]()          // padded 16 s audio per chunk (for embeddings)
        binarized.reserveCapacity(starts.count)
        windows.reserveCapacity(starts.count)

        timings.segmentation = 0
        for s in starts {
            var window = [Float](repeating: 0, count: windowSamples)
            let count = Swift.min(windowSamples, n - s)
            if count > 0 { for i in 0..<count { window[i] = waveform[s + i] } }
            windows.append(window)

            let tSeg = Date()
            let seg = try segmentation.segment(chunk: window)      // [799][11] log-probs
            timings.segmentation += Date().timeIntervalSince(tSeg)
            var multilabel = powerset.toMultilabel(seg.frames)     // [799][numSpeakers] 0/1
            Self.medianFilterFrames(&multilabel, window: Self.medianWindow)
            binarized.append(multilabel)
        }

        let tEmb = Date()
        let embeddings = try extractEmbeddings(binarized: binarized, windows: windows, numSpeakers: numSpeakers)
        timings.embedding = Date().timeIntervalSince(tEmb)
        return DiarizationFeatures(
            binarized: binarized,
            embeddings: embeddings,
            window: SlidingWindow(start: 0, duration: Double(windowSamples) / Double(sampleRate), step: Self.stepSeconds)
        )
    }

    /// pyannote get_embeddings with exclude_overlap=True: zero out frames with ≥2
    /// active speakers, use that "clean" mask per speaker unless it is too short
    /// (≤ minNumFrames), else fall back to the full mask.
    private func extractEmbeddings(
        binarized: [[[Float]]], windows: [[Float]], numSpeakers: Int
    ) throws -> [[[Float]]] {
        var result = [[[Float]]]()
        result.reserveCapacity(binarized.count)

        for (c, chunk) in binarized.enumerated() {
            let numFrames = chunk.count
            // clean_frames: 1 where fewer than 2 speakers active, else 0.
            var clean = [Bool](repeating: false, count: numFrames)
            for f in 0..<numFrames {
                var active = 0
                for spk in 0..<numSpeakers where chunk[f][spk] > 0.5 { active += 1 }
                clean[f] = active < 2
            }

            var chunkEmb = [[Float]]()
            chunkEmb.reserveCapacity(numSpeakers)
            for spk in 0..<numSpeakers {
                var mask = [Float](repeating: 0, count: numFrames)
                var cleanMask = [Float](repeating: 0, count: numFrames)
                var cleanSum: Float = 0
                for f in 0..<numFrames {
                    let v = chunk[f][spk]
                    mask[f] = v
                    let cv = clean[f] ? v : 0
                    cleanMask[f] = cv
                    cleanSum += cv
                }
                let used = cleanSum > Float(Self.minNumFrames) ? cleanMask : mask
                chunkEmb.append(try embeddings.embed(region: windows[c], weights: used))
            }
            result.append(chunkEmb)
        }
        return result
    }

    /// scipy `median_filter(size=(1, window, 1), mode="reflect")` over the frame axis,
    /// per speaker. `frames[f][speaker]` in place.
    static func medianFilterFrames(_ frames: inout [[Float]], window: Int) {
        let n = frames.count
        guard n > 0, window > 1 else { return }
        let numSpk = frames[0].count
        let radius = window / 2
        let original = frames
        for spk in 0..<numSpk {
            for f in 0..<n {
                var vals = [Float](repeating: 0, count: window)
                for j in 0..<window {
                    let idx = reflectIndex(f - radius + j, n)
                    vals[j] = original[idx][spk]
                }
                vals.sort()
                frames[f][spk] = vals[window / 2]  // odd window → middle element
            }
        }
    }

    /// Half-sample-symmetric reflection (scipy "reflect"): index -1→0, n→n-1.
    static func reflectIndex(_ i: Int, _ n: Int) -> Int {
        var idx = i
        while idx < 0 || idx >= n {
            if idx < 0 { idx = -idx - 1 }
            if idx >= n { idx = 2 * n - idx - 1 }
        }
        return idx
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
