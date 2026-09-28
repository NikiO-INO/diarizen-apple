import Foundation

/// Powerset → multi-label decoding, matching pyannote's `Powerset` module.
///
/// The segmentation head emits `numPowersetClasses` log-probs per frame, each class
/// a *subset* of speakers (size 0…maxSetSize). For base-s80-md that's 11 classes
/// over 4 speakers (subsets of size ≤ 2). Hard decoding (`soft=False`) is:
/// per frame, argmax the powerset logits, then map that class to its speaker set.
public struct Powerset {
    public let numSpeakers: Int          // multi-label classes (4)
    public let maxSetSize: Int           // max simultaneous speakers (2)
    public let numPowersetClasses: Int   // (11)
    /// `mapping[k]` = the 0/1 speaker vector of powerset class `k`.
    let mapping: [[Float]]

    public init(numSpeakers: Int = 4, maxSetSize: Int = 2) {
        self.numSpeakers = numSpeakers
        self.maxSetSize = maxSetSize
        let m = Self.buildMapping(numSpeakers: numSpeakers, maxSetSize: maxSetSize)
        self.mapping = m
        self.numPowersetClasses = m.count
    }

    /// Infer `maxSetSize` from the segmentation head's class count, so the same code
    /// serves base (11 classes → max 2) and large-s80-md-v2 (16 classes → max 4).
    /// DiariZen models use 4 local speakers; `numPowersetClasses = Σ_{k≤maxSetSize} C(4,k)`.
    public init(numSpeakers: Int = 4, numPowersetClasses: Int) {
        var acc = 0
        var maxSet = numSpeakers
        for k in 0...numSpeakers {
            acc += Self.binomial(numSpeakers, k)
            if acc == numPowersetClasses { maxSet = k; break }
        }
        self.init(numSpeakers: numSpeakers, maxSetSize: maxSet)
        precondition(self.numPowersetClasses == numPowersetClasses,
                     "no maxSetSize for \(numSpeakers) speakers gives \(numPowersetClasses) powerset classes")
    }

    static func binomial(_ n: Int, _ k: Int) -> Int {
        if k < 0 || k > n { return 0 }
        var r = 1
        for i in 0..<min(k, n - k) { r = r * (n - i) / (i + 1) }
        return r
    }

    /// Hard-decode powerset log-probs `[frames][numPowersetClasses]` to multi-label
    /// speaker activity `[frames][numSpeakers]` (0/1), matching `to_multilabel(soft=False)`.
    public func toMultilabel(_ powerset: [[Float]]) -> [[Float]] {
        powerset.map { frame in
            var best = 0
            var bestVal = frame.isEmpty ? 0 : frame[0]
            for k in 1..<frame.count where frame[k] > bestVal { bestVal = frame[k]; best = k }
            return mapping[best]
        }
    }

    /// Build the `(numPowersetClasses, numSpeakers)` mapping in pyannote's order:
    /// increasing set size, `combinations(range(numSpeakers), size)` within each size.
    static func buildMapping(numSpeakers: Int, maxSetSize: Int) -> [[Float]] {
        var rows = [[Float]]()
        for size in 0...maxSetSize {
            for combo in combinations(numSpeakers, size) {
                var row = [Float](repeating: 0, count: numSpeakers)
                for s in combo { row[s] = 1 }
                rows.append(row)
            }
        }
        return rows
    }

    /// `combinations(range(n), k)` in lexicographic order (matches itertools).
    static func combinations(_ n: Int, _ k: Int) -> [[Int]] {
        if k == 0 { return [[]] }
        if k > n { return [] }
        var result = [[Int]]()
        var current = [Int]()
        current.reserveCapacity(k)
        func choose(_ start: Int) {
            if current.count == k { result.append(current); return }
            // enough remaining elements to still fill k
            let last = n - (k - current.count)
            var v = start
            while v <= last {
                current.append(v)
                choose(v + 1)
                current.removeLast()
                v += 1
            }
        }
        choose(0)
        return result
    }
}
