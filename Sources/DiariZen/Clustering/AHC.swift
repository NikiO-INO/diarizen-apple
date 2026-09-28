import Foundation

/// Agglomerative hierarchical clustering — centroid linkage (UPGMC) + a
/// distance-criterion flat cut, matching scipy's `linkage(method="centroid")`
/// + `fcluster(criterion="distance")` as used to initialize VBx.
///
/// Centroid linkage merges the two clusters whose (size-weighted) mean vectors are
/// closest in Euclidean distance. Inputs are the L2-normalized embeddings. Sizes
/// are tiny (≤ a few dozen), so the naive O(n³) merge loop is fine.
public enum AHC {
    /// One dendrogram merge: the two node ids joined, their centroid distance, and
    /// the merged size. Leaf nodes are `0..<n`; merge k creates node `n + k`.
    public struct Merge { public let a: Int; public let b: Int; public let dist: Double; public let size: Int }

    /// O(n²·d) via a nearest-neighbor cache (vs the naive O(n³·d)). Produces the
    /// identical dendrogram — each step still merges the globally-closest centroid
    /// pair — but scales to the thousands of embeddings a long meeting yields.
    /// The `if dist(k, merged) < nnDist[k]` update handles centroid linkage's
    /// non-monotonicity (a merged centroid can be closer than either child was).
    public static func centroidLinkage(_ points: [[Double]]) -> [Merge] {
        let n = points.count
        guard n > 1 else { return [] }
        let dim = points[0].count
        var centroid = points
        var size = [Int](repeating: 1, count: n)
        var active = [Bool](repeating: true, count: n)
        var nodeId = Array(0..<n)
        var nextId = n
        var merges = [Merge]()
        merges.reserveCapacity(n - 1)

        var nn = [Int](repeating: -1, count: n)          // nearest active index
        var nnDist = [Double](repeating: .infinity, count: n)

        func dist(_ a: Int, _ b: Int) -> Double { euclidean(centroid[a], centroid[b]) }
        func recomputeNN(_ i: Int) {
            var bd = Double.infinity, bj = -1
            for j in 0..<n where active[j] && j != i {
                let d = dist(i, j)
                if d < bd { bd = d; bj = j }
            }
            nn[i] = bj; nnDist[i] = bd
        }
        for i in 0..<n { recomputeNN(i) }

        for _ in 0..<(n - 1) {
            // Globally-closest pair = the active slot with the smallest NN distance.
            var i = -1, bd = Double.infinity
            for k in 0..<n where active[k] {
                if nnDist[k] < bd { bd = nnDist[k]; i = k }
            }
            let j = nn[i]
            let si = size[i], sj = size[j], sij = si + sj
            var merged = [Double](repeating: 0, count: dim)
            for k in 0..<dim {
                merged[k] = (Double(si) * centroid[i][k] + Double(sj) * centroid[j][k]) / Double(sij)
            }
            merges.append(Merge(a: nodeId[i], b: nodeId[j], dist: bd, size: sij))
            centroid[i] = merged; size[i] = sij; nodeId[i] = nextId; nextId += 1
            active[j] = false

            recomputeNN(i)
            for k in 0..<n where active[k] && k != i {
                if nn[k] == i || nn[k] == j {
                    recomputeNN(k)                       // its NN was one of the merged pair
                } else {
                    let d = dist(k, i)                   // merged centroid may now be closer
                    if d < nnDist[k] { nnDist[k] = d; nn[k] = i }
                }
            }
        }
        return merges
    }

    /// Flat clusters by cutting the dendrogram at `threshold` on the cophenetic
    /// (max in-subtree merge) distance — handles centroid linkage's non-monotonicity.
    /// Returns a 0-based label per original point, labels assigned in cut order.
    public static func fclusterDistance(_ merges: [Merge], n: Int, threshold: Double) -> [Int] {
        guard n > 1 else { return [Int](repeating: 0, count: n) }
        let numNodes = 2 * n - 1
        var coph = [Double](repeating: 0, count: numNodes)   // leaves = 0
        var left = [Int](repeating: -1, count: numNodes)
        var right = [Int](repeating: -1, count: numNodes)
        for (k, m) in merges.enumerated() {
            let node = n + k
            left[node] = m.a; right[node] = m.b
            coph[node] = Swift.max(m.dist, coph[m.a], coph[m.b])
        }

        var labels = [Int](repeating: -1, count: n)
        var nextLabel = 0
        func assignAll(_ node: Int, _ label: Int) {
            if node < n { labels[node] = label; return }
            assignAll(left[node], label); assignAll(right[node], label)
        }
        func cut(_ node: Int) {
            if node < n { labels[node] = nextLabel; nextLabel += 1; return }
            if coph[node] <= threshold { assignAll(node, nextLabel); nextLabel += 1 }
            else { cut(left[node]); cut(right[node]) }
        }
        cut(numNodes - 1)
        return labels
    }
}

func euclidean(_ a: [Double], _ b: [Double]) -> Double {
    var s = 0.0
    for i in a.indices { let d = a[i] - b[i]; s += d * d }
    return s.squareRoot()
}
