import Foundation

/// Maximum-weight assignment, matching `scipy.optimize.linear_sum_assignment(maximize=True)`
/// for the constrained speaker→cluster assignment. Matches `min(rows, cols)` pairs,
/// each row and column used at most once, maximizing the total weight.
///
/// Cost matrices here are tiny (num_speakers ≤ 4), so an exact branch over the
/// smaller dimension is both simple and fast.
public enum Hungarian {
    /// Returns `(row, col)` pairs of the optimal assignment, size `min(rows, cols)`.
    public static func maxAssignment(_ cost: [[Double]]) -> [(Int, Int)] {
        let r = cost.count
        let c = cost.first?.count ?? 0
        if r == 0 || c == 0 { return [] }

        var bestSum = -Double.infinity
        var bestPairs = [(Int, Int)]()
        var current = [(Int, Int)]()

        if r <= c {
            var usedCol = [Bool](repeating: false, count: c)
            func rec(_ row: Int, _ sum: Double) {
                if row == r {
                    if sum > bestSum { bestSum = sum; bestPairs = current }
                    return
                }
                for col in 0..<c where !usedCol[col] {
                    usedCol[col] = true; current.append((row, col))
                    rec(row + 1, sum + cost[row][col])
                    current.removeLast(); usedCol[col] = false
                }
            }
            rec(0, 0)
        } else {
            var usedRow = [Bool](repeating: false, count: r)
            func rec(_ col: Int, _ sum: Double) {
                if col == c {
                    if sum > bestSum { bestSum = sum; bestPairs = current }
                    return
                }
                for row in 0..<r where !usedRow[row] {
                    usedRow[row] = true; current.append((row, col))
                    rec(col + 1, sum + cost[row][col])
                    current.removeLast(); usedRow[row] = false
                }
            }
            rec(0, 0)
        }
        return bestPairs
    }
}
