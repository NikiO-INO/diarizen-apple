import Foundation

/// DiariZen's VBx clustering: map per-(chunk, speaker) embeddings to global speaker
/// ids. Ports `VBxClustering.__call__` — filter → AHC init → PLDA+VBx → cluster
/// centroids → cosine assignment (constrained) → renumber. All CPU/Swift.
public struct VBxClustering {
    public let plda: PLDA
    public var fa: Double
    public var fb: Double
    public var ahcThreshold: Double
    public var maxIters: Int
    public var minFramesRatio: Double
    public var constrained: Bool
    public var spEps: Double  // keep VBx clusters with prior > spEps

    public init(
        plda: PLDA, fa: Double = 0.07, fb: Double = 0.8, ahcThreshold: Double = 0.6,
        maxIters: Int = 20, minFramesRatio: Double = 0.1, constrained: Bool = true,
        spEps: Double = 1e-7
    ) {
        self.plda = plda
        self.fa = fa; self.fb = fb; self.ahcThreshold = ahcThreshold
        self.maxIters = maxIters; self.minFramesRatio = minFramesRatio
        self.constrained = constrained; self.spEps = spEps
    }

    /// `embeddings[chunk][speaker][dim]`, `binarized[chunk][frame][speaker]` →
    /// `hard_clusters[chunk][speaker]` (cluster id; -2 = unassigned, renumbered).
    public func cluster(embeddings: [[[Float]]], binarized: [[[Float]]]) -> [[Int]] {
        let numChunks = embeddings.count
        let numSpeakers = embeddings.first?.count ?? 0
        let dim = embeddings.first?.first?.count ?? 0

        // 1. filter_embeddings: active & (≥ minFrames clean frames). C-order.
        let (trainEmb, _, _) = filterEmbeddings(embeddings: embeddings, binarized: binarized)
        if trainEmb.count < 2 {
            return [[Int]](repeating: [Int](repeating: 0, count: numSpeakers), count: numChunks)
        }

        // 2. AHC (centroid linkage on L2-normalized embeddings, distance cut).
        let normed = trainEmb.map { row -> [Double] in
            var v = row.map(Double.init); l2Normalize(&v); return v
        }
        let merges = AHC.centroidLinkage(normed)
        let ahc = renumber(AHC.fclusterDistance(merges, n: trainEmb.count, threshold: ahcThreshold))

        // 3. PLDA transform + VBx.
        let fea = plda.transform(trainEmb)
        let (gamma, pi) = VBx.cluster(
            ahcInit: ahc, fea: fea, phi: plda.phi,
            Fa: fa, Fb: fb, maxIters: maxIters)

        // 4. centroids = q[:, sp>eps]ᵀ · train_embeddings  (raw 256-d, unnormalized).
        let keptCols = (0..<pi.count).filter { pi[$0] > spEps }
        var centroids = [[Double]](repeating: [Double](repeating: 0, count: dim), count: keptCols.count)
        for (ki, k) in keptCols.enumerated() {
            for t in 0..<trainEmb.count {
                let w = gamma[t][k]
                if w == 0 { continue }
                let e = trainEmb[t]
                for d in 0..<dim { centroids[ki][d] += w * Double(e[d]) }
            }
        }

        // 5-6. soft = 2 − cosine_distance; assign each (chunk, speaker) to a cluster.
        var hard = [[Int]](repeating: [Int](repeating: -2, count: numSpeakers), count: numChunks)
        for c in 0..<numChunks {
            var cost = [[Double]](repeating: [Double](repeating: 0, count: centroids.count), count: numSpeakers)
            for s in 0..<numSpeakers {
                let emb = embeddings[c][s].map(Double.init)
                for ki in 0..<centroids.count {
                    cost[s][ki] = 2.0 - cosineDistance(emb, centroids[ki])
                }
            }
            if constrained {
                for (s, k) in Hungarian.maxAssignment(cost) { hard[c][s] = k }
            } else {
                for s in 0..<numSpeakers { hard[c][s] = argmax(cost[s]) }
            }
        }

        // 7. renumber via unique-inverse (matches np.unique(return_inverse)).
        return renumber2D(hard)
    }

    /// active (any frame) & clean-frame-count ≥ round(minFramesRatio·numFrames).
    /// Returns train embeddings in (chunk, speaker) C-order + their origins.
    func filterEmbeddings(
        embeddings: [[[Float]]], binarized: [[[Float]]]
    ) -> (train: [[Float]], chunkIdx: [Int], speakerIdx: [Int]) {
        let numChunks = binarized.count
        let numFrames = binarized.first?.count ?? 0
        let numSpeakers = embeddings.first?.count ?? 0
        let minFrames = Int((minFramesRatio * Double(numFrames)).rounded())

        var train = [[Float]](); var cIdx = [Int](); var sIdx = [Int]()
        for c in 0..<numChunks {
            // clean frame = exactly one speaker active
            var singleActive = [Bool](repeating: false, count: numFrames)
            for f in 0..<numFrames {
                var active = 0
                for s in 0..<numSpeakers where binarized[c][f][s] > 0.5 { active += 1 }
                singleActive[f] = active == 1
            }
            for s in 0..<numSpeakers {
                var total = 0, clean = 0
                for f in 0..<numFrames where binarized[c][f][s] > 0.5 {
                    total += 1
                    if singleActive[f] { clean += 1 }
                }
                if total > 0 && clean >= minFrames {  // active & enough clean frames
                    train.append(embeddings[c][s]); cIdx.append(c); sIdx.append(s)
                }
            }
        }
        return (train, cIdx, sIdx)
    }
}

// MARK: - small numeric helpers

/// Cosine distance = 1 − (a·b)/(‖a‖‖b‖), matching scipy cdist(metric="cosine").
func cosineDistance(_ a: [Double], _ b: [Double]) -> Double {
    var dot = 0.0, na = 0.0, nb = 0.0
    for i in a.indices { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
    let denom = (na.squareRoot() * nb.squareRoot())
    return denom > 0 ? 1.0 - dot / denom : 1.0
}

func argmax(_ v: [Double]) -> Int {
    var best = 0
    for i in 1..<v.count where v[i] > v[best] { best = i }
    return best
}

/// np.unique(x, return_inverse=True): sort unique values, map each to its rank.
func renumber(_ labels: [Int]) -> [Int] {
    let uniq = Array(Set(labels)).sorted()
    var rank = [Int: Int](); for (i, u) in uniq.enumerated() { rank[u] = i }
    return labels.map { rank[$0]! }
}

func renumber2D(_ m: [[Int]]) -> [[Int]] {
    let uniq = Array(Set(m.flatMap { $0 })).sorted()
    var rank = [Int: Int](); for (i, u) in uniq.enumerated() { rank[u] = i }
    return m.map { $0.map { rank[$0]! } }
}
