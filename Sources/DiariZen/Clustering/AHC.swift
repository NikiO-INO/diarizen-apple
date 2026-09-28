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

    public static func centroidLinkage(_ points: [[Double]]) -> [Merge] {
        let n = points.count
        guard n > 1 else { return [] }
        let dim = points[0].count
        var centroid = points
        var size = [Int](repeating: 1, count: n)
        var active = [Bool](repeating: true, count: n)
        var nodeId = Array(0..<n)   // current dendrogram node id per slot
        var merges = [Merge]()
        var nextId = n

        for _ in 0..<(n - 1) {
            var bestD = Double.infinity, bi = -1, bj = -1
            for i in 0..<n where active[i] {
                let ci = centroid[i]
                for j in (i + 1)..<n where active[j] {
                    let d = euclidean(ci, centroid[j])
                    if d < bestD { bestD = d; bi = i; bj = j }
                }
            }
            let si = size[bi], sj = size[bj], sij = si + sj
            var merged = [Double](repeating: 0, count: dim)
            for k in 0..<dim {
                merged[k] = (Double(si) * centroid[bi][k] + Double(sj) * centroid[bj][k]) / Double(sij)
            }
            merges.append(Merge(a: nodeId[bi], b: nodeId[bj], dist: bestD, size: sij))
            centroid[bi] = merged
            size[bi] = sij
            nodeId[bi] = nextId
            active[bj] = false
            nextId += 1
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
