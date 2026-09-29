import Foundation

/// VBx (Bayesian HMM x-vector clustering), the `loopProb = 0` GMM variant DiariZen
/// uses (`cluster_vbx`). Refines an AHC initialization into soft speaker
/// responsibilities `gamma` (T×S) and priors `pi` (S); redundant speakers' priors
/// decay toward zero, which is how the final speaker count is decided.
///
/// Ported from `diarizen.clustering.VBx.VBx`; equation numbers refer to
/// Landini et al., "Bayesian HMM clustering of x-vector sequences (VBx)".
/// Double precision throughout to match numpy.
public enum VBx {
    /// - Parameters:
    ///   - ahcInit: length-T cluster labels (0-based) from AHC.
    ///   - fea: T×D PLDA-space features.
    ///   - phi: D across-class covariance diagonal.
    public static func cluster(
        ahcInit: [Int], fea: [[Double]], phi: [Double],
        Fa: Double, Fb: Double, maxIters: Int,
        initSmoothing: Double = 7.0, epsilon: Double = 1e-4
    ) -> (gamma: [[Double]], pi: [Double]) {
        let T = fea.count
        let D = phi.count
        let S = (ahcInit.max() ?? -1) + 1
        precondition(T > 0 && S > 0)

        // qinit: softmax(one_hot(ahc) * init_smoothing) per row.
        var gamma = [[Double]](repeating: [Double](repeating: 0, count: S), count: T)
        for t in 0..<T {
            var row = [Double](repeating: 0, count: S)
            row[ahcInit[t]] = initSmoothing
            softmaxInPlace(&row)
            gamma[t] = row
        }
        var pi = [Double](repeating: 1.0 / Double(S), count: S)  // VBx pi=int → ones/S

        // Per-frame constants: rho = X·√Phi (18), G (23 constant term).
        let V = phi.map { $0.squareRoot() }
        var rho = [[Double]](repeating: [Double](repeating: 0, count: D), count: T)
        var G = [Double](repeating: 0, count: T)
        let dLog2pi = Double(D) * Foundation.log(2 * Double.pi)
        for t in 0..<T {
            var ss = 0.0
            for d in 0..<D { rho[t][d] = fea[t][d] * V[d]; ss += fea[t][d] * fea[t][d] }
            G[t] = -0.5 * (ss + dLog2pi)
        }

        let ratio = Fa / Fb
        var prevELBO = 0.0
        for iter in 0..<maxIters {
            // gamma column sums, gamma.T·rho.
            var gammaSum = [Double](repeating: 0, count: S)
            var gRho = [[Double]](repeating: [Double](repeating: 0, count: D), count: S)
            for t in 0..<T {
                let gt = gamma[t]
                for s in 0..<S where gt[s] != 0 {
                    gammaSum[s] += gt[s]
                    let g = gt[s], rt = rho[t]
                    for d in 0..<D { gRho[s][d] += g * rt[d] }
                }
            }

            // invL (17), alpha (16) per speaker.
            var invL = [[Double]](repeating: [Double](repeating: 0, count: D), count: S)
            var alpha = [[Double]](repeating: [Double](repeating: 0, count: D), count: S)
            var halfTerm = [Double](repeating: 0, count: S)  // 0.5·(invL+alpha²)·Phi
            for s in 0..<S {
                var h = 0.0
                for d in 0..<D {
                    let il = 1.0 / (1.0 + ratio * gammaSum[s] * phi[d])
                    let al = ratio * il * gRho[s][d]
                    invL[s][d] = il; alpha[s][d] = al
                    h += (il + al * al) * phi[d]
                }
                halfTerm[s] = 0.5 * h
            }

            // log_p_ (23) + log pi, responsibilities (GMM path), pi update.
            let lpi = pi.map { Foundation.log($0 + 1e-8) }
            var totalLogPx = 0.0
            for t in 0..<T {
                var logp = [Double](repeating: 0, count: S)
                let rt = rho[t], gt = G[t]
                for s in 0..<S {
                    var dot = 0.0
                    let al = alpha[s]
                    for d in 0..<D { dot += rt[d] * al[d] }
                    logp[s] = Fa * (dot - halfTerm[s] + gt) + lpi[s]
                }
                let logPx = logSumExp(logp)
                totalLogPx += logPx
                for s in 0..<S { gamma[t][s] = Foundation.exp(logp[s] - logPx) }
            }
            var piSum = 0.0
            for s in 0..<S { var ps = 0.0; for t in 0..<T { ps += gamma[t][s] }; pi[s] = ps; piSum += ps }
            for s in 0..<S { pi[s] /= piSum }

            // ELBO (25) with early stop.
            var reg = 0.0
            for s in 0..<S {
                for d in 0..<D {
                    let il = invL[s][d], al = alpha[s][d]
                    reg += Foundation.log(il) - il - al * al + 1
                }
            }
            let elbo = totalLogPx + Fb * 0.5 * reg
            if iter > 0 && elbo - prevELBO < epsilon { break }
            prevELBO = elbo
        }
        return (gamma, pi)
    }
}

func softmaxInPlace(_ v: inout [Double]) {
    guard let m = v.max() else { return }
    var sum = 0.0
    for i in v.indices { v[i] = Foundation.exp(v[i] - m); sum += v[i] }
    if sum > 0 { for i in v.indices { v[i] /= sum } }
}

func logSumExp(_ v: [Double]) -> Double {
    guard let m = v.max() else { return -Double.infinity }
    var sum = 0.0
    for x in v { sum += Foundation.exp(x - m) }
    return m + Foundation.log(sum)
}
