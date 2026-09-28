import Foundation

/// PLDA x-vector transform for VBx, from constants exported by
/// `conversion/export_plda.py` (which folds in the generalized eigendecomposition,
/// so Swift needs only linear algebra). Maps a 256-d speaker embedding into the
/// 128-d PLDA space where VBx clusters. Double precision throughout to match numpy.
///
/// Reproduces `diarizen.clustering.VBx.vbx_setup`'s `xvec_tf` ∘ `plda_tf`:
///   xvec_tf(x) = √L · l2( (lda·ᵀ (√D · l2(x−mean1))) − mean2 )
///   plda_tf(y) = (y − plda_mu) · plda_trᵀ            (lda_dim = 128, no truncation)
public struct PLDA {
    let mean1: [Double]      // (256)
    let mean2: [Double]      // (128)
    let lda: [[Double]]      // (256, 128)
    let pldaMu: [Double]     // (128)
    let pldaTr: [[Double]]   // (128, 128)
    public let phi: [Double] // (128) across-class covariance diagonal (VBx Phi)

    let inDim: Int           // 256
    let ldaDim: Int          // 128

    public init(contentsOf url: URL) throws {
        let raw = try JSONDecoder().decode([String: Field].self, from: Data(contentsOf: url))
        func vec(_ k: String) throws -> [Double] {
            guard let f = raw[k] else { throw DiariZenError.modelLoad("plda transform missing '\(k)'") }
            return f.data
        }
        func mat(_ k: String) throws -> [[Double]] {
            guard let f = raw[k], f.shape.count == 2 else {
                throw DiariZenError.modelLoad("plda transform missing/!2D '\(k)'")
            }
            let (r, c) = (f.shape[0], f.shape[1])
            return (0..<r).map { Array(f.data[$0 * c ..< ($0 + 1) * c]) }
        }
        self.mean1 = try vec("mean1")
        self.mean2 = try vec("mean2")
        self.lda = try mat("lda")
        self.pldaMu = try vec("plda_mu")
        self.pldaTr = try mat("plda_tr")
        self.phi = try vec("plda_psi")
        self.inDim = mean1.count
        self.ldaDim = mean2.count
    }

    struct Field: Decodable { let shape: [Int]; let data: [Double] }

    /// Transform raw embeddings `[N][256]` (Float) → PLDA features `[N][128]` (Double).
    public func transform(_ embeddings: [[Float]]) -> [[Double]] {
        embeddings.map { pldaStep(xvecStep($0.map(Double.init))) }
    }

    /// xvec_tf for one embedding.
    private func xvecStep(_ x: [Double]) -> [Double] {
        // √D · l2(x − mean1)
        var centered = [Double](repeating: 0, count: inDim)
        for i in 0..<inDim { centered[i] = x[i] - mean1[i] }
        l2Normalize(&centered)
        let sqrtD = (Double(inDim)).squareRoot()
        for i in 0..<inDim { centered[i] *= sqrtD }

        // ldaᵀ · centered  →  (128); lda is (256,128) so column j = Σ_i lda[i][j]·centered[i]
        var y = [Double](repeating: 0, count: ldaDim)
        for i in 0..<inDim {
            let ci = centered[i]
            if ci == 0 { continue }
            let row = lda[i]
            for j in 0..<ldaDim { y[j] += row[j] * ci }
        }
        // − mean2, then √L · l2(·)
        for j in 0..<ldaDim { y[j] -= mean2[j] }
        l2Normalize(&y)
        let sqrtL = (Double(ldaDim)).squareRoot()
        for j in 0..<ldaDim { y[j] *= sqrtL }
        return y
    }

    /// plda_tf for one xvec-transformed vector: (y − mu) · pldaTrᵀ.
    private func pldaStep(_ y: [Double]) -> [Double] {
        var d = [Double](repeating: 0, count: ldaDim)
        for j in 0..<ldaDim { d[j] = y[j] - pldaMu[j] }
        // row r of pldaTr, output[r] = Σ_j pldaTr[r][j]·d[j]
        var out = [Double](repeating: 0, count: ldaDim)
        for r in 0..<ldaDim {
            let row = pldaTr[r]
            var s = 0.0
            for j in 0..<ldaDim { s += row[j] * d[j] }
            out[r] = s
        }
        return out
    }
}

func l2Normalize(_ v: inout [Double]) {
    var n = 0.0
    for x in v { n += x * x }
    n = n.squareRoot()
    if n > 0 { for i in v.indices { v[i] /= n } }
}
