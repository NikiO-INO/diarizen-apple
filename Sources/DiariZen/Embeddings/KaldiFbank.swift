import Accelerate
import Foundation

/// Kaldi-compatible log-mel filterbank frontend, native Swift (Accelerate/vDSP).
///
/// Reproduces `torchaudio.compliance.kaldi.fbank` with DiariZen/WeSpeaker's exact
/// options — 80 mel bins, 25/10 ms Hamming frames, preemphasis 0.97, DC removal,
/// power spectrum, log — followed by the per-utterance CMVN (`- mean over frames`)
/// that `WeSpeakerResNet34.compute_fbank` applies. Deterministic (`dither = 0`).
///
/// This is the frontend DiariZen owns; only the learned ResNet34 lives in CoreML.
/// Semantics are validated against kaldi in `validation/compare_swift_fbank.py`.
public struct KaldiFbank {
    public let sampleRate: Int
    public let numMelBins: Int
    let frameLength: Int      // samples per frame (400 @ 25 ms / 16 kHz)
    let frameShift: Int       // hop (160 @ 10 ms)
    let fftSize: Int          // padded window, next pow2 (512)
    let numFftBins: Int       // fftSize/2 (256) — kaldi drops the Nyquist bin
    let preemph: Float
    let waveformScale: Float  // WeSpeaker scales [-1,1] float to int16 range

    private let window: [Float]           // Hamming, length frameLength
    private let melMatrix: [[Float]]       // [numMelBins][numFftBins]
    private let dft: vDSP.DFT<Float>

    public init(
        sampleRate: Int = 16_000,
        numMelBins: Int = 80,
        frameLengthMs: Double = 25.0,
        frameShiftMs: Double = 10.0,
        lowFreq: Double = 20.0,
        highFreq: Double = 0.0,       // 0 → Nyquist
        preemphasis: Float = 0.97,
        waveformScale: Float = 32768.0
    ) {
        self.sampleRate = sampleRate
        self.numMelBins = numMelBins
        self.frameLength = Int((frameLengthMs / 1000.0) * Double(sampleRate))
        self.frameShift = Int((frameShiftMs / 1000.0) * Double(sampleRate))
        self.preemph = preemphasis
        self.waveformScale = waveformScale

        var padded = 1
        while padded < frameLength { padded <<= 1 }
        self.fftSize = padded
        self.numFftBins = padded / 2

        self.window = Self.hammingWindow(frameLength)
        let nyquist = highFreq > 0 ? highFreq : Double(sampleRate) / 2.0
        self.melMatrix = Self.melFilterbank(
            numBins: numMelBins, numFftBins: padded / 2, fftSize: padded,
            sampleRate: Double(sampleRate), lowFreq: lowFreq, highFreq: nyquist
        )
        self.dft = vDSP.DFT(count: padded, direction: .forward,
                            transformType: .complexComplex, ofType: Float.self)!
    }

    /// Compute the CMVN'd log-mel features for a mono waveform in [-1, 1].
    /// Returns `[numFrames][numMelBins]`, matching `compute_fbank`.
    public func compute(_ waveform: [Float]) -> [[Float]] {
        let n = waveform.count
        if n < frameLength { return [] }
        let numFrames = 1 + (n - frameLength) / frameShift  // kaldi snip_edges

        var scaled = [Float](repeating: 0, count: n)
        vDSP.multiply(waveformScale, waveform, result: &scaled)

        var frames = [[Float]]()
        frames.reserveCapacity(numFrames)
        for f in 0..<numFrames {
            frames.append(logMelFrame(scaled, start: f * frameShift))
        }
        applyCMVN(&frames)
        return frames
    }

    // One frame: DC removal → preemphasis → Hamming → pad → power spectrum → mel → log.
    private func logMelFrame(_ signal: [Float], start: Int) -> [Float] {
        var frame = Array(signal[start..<(start + frameLength)])

        // remove_dc_offset: subtract frame mean
        let mean = vDSP.mean(frame)
        vDSP.add(-mean, frame, result: &frame)

        // preemphasis with replicate padding (x[-1] = x[0]); do it back-to-front.
        if preemph != 0 {
            var prev = frame[0]  // x[-1] replicate
            for i in 0..<frameLength {
                let cur = frame[i]
                frame[i] = cur - preemph * prev
                prev = cur
            }
        }

        // Hamming window
        vDSP.multiply(frame, window, result: &frame)

        // Zero-pad to fftSize, forward complex DFT (imag = 0)
        var realIn = frame + [Float](repeating: 0, count: fftSize - frameLength)
        var imagIn = [Float](repeating: 0, count: fftSize)
        var realOut = [Float](repeating: 0, count: fftSize)
        var imagOut = [Float](repeating: 0, count: fftSize)
        dft.transform(inputReal: realIn, inputImaginary: imagIn,
                      outputReal: &realOut, outputImaginary: &imagOut)

        // Power spectrum on bins 0..<numFftBins (use_power = True)
        var power = [Float](repeating: 0, count: numFftBins)
        for k in 0..<numFftBins {
            power[k] = realOut[k] * realOut[k] + imagOut[k] * imagOut[k]
        }

        // Mel filterbank + log(max(·, eps))
        var out = [Float](repeating: 0, count: numMelBins)
        let eps = Float.ulpOfOne
        for m in 0..<numMelBins {
            var e: Float = 0
            let row = melMatrix[m]
            for k in 0..<numFftBins { e += row[k] * power[k] }
            out[m] = Foundation.log(Swift.max(e, eps))
        }
        _ = realIn  // silence unused-mutation warning
        return out
    }

    // Per-utterance CMVN: subtract each mel bin's mean over frames (compute_fbank).
    private func applyCMVN(_ frames: inout [[Float]]) {
        guard let first = frames.first else { return }
        let numFrames = frames.count
        let bins = first.count
        var means = [Float](repeating: 0, count: bins)
        for row in frames { for b in 0..<bins { means[b] += row[b] } }
        for b in 0..<bins { means[b] /= Float(numFrames) }
        for i in 0..<numFrames { for b in 0..<bins { frames[i][b] -= means[b] } }
    }

    // MARK: - Coefficient tables

    static func hammingWindow(_ n: Int) -> [Float] {
        let a = 2.0 * Double.pi / Double(n - 1)
        return (0..<n).map { Float(0.54 - 0.46 * cos(a * Double($0))) }
    }

    /// Kaldi mel scale: 1127 * ln(1 + f/700).
    static func mel(_ f: Double) -> Double { 1127.0 * Foundation.log(1.0 + f / 700.0) }

    /// Triangular mel filterbank matching kaldi `MelBanks`.
    static func melFilterbank(
        numBins: Int, numFftBins: Int, fftSize: Int,
        sampleRate: Double, lowFreq: Double, highFreq: Double
    ) -> [[Float]] {
        let fftBinWidth = sampleRate / Double(fftSize)
        let melLow = mel(lowFreq)
        let melHigh = mel(highFreq)
        let melDelta = (melHigh - melLow) / Double(numBins + 1)

        var banks = [[Float]](repeating: [Float](repeating: 0, count: numFftBins), count: numBins)
        for m in 0..<numBins {
            let leftMel = melLow + Double(m) * melDelta
            let centerMel = melLow + Double(m + 1) * melDelta
            let rightMel = melLow + Double(m + 2) * melDelta
            for k in 0..<numFftBins {
                let freq = fftBinWidth * Double(k)
                let melK = mel(freq)
                if melK > leftMel && melK < rightMel {
                    let w = melK <= centerMel
                        ? (melK - leftMel) / (centerMel - leftMel)
                        : (rightMel - melK) / (rightMel - centerMel)
                    banks[m][k] = Float(w)
                }
            }
        }
        return banks
    }
}
