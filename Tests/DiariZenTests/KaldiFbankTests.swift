import XCTest
@testable import DiariZen

final class KaldiFbankTests: XCTestCase {
    func testHammingWindow() {
        let w = KaldiFbank.hammingWindow(400)
        XCTAssertEqual(w.count, 400)
        // kaldi hamming: 0.54 - 0.46*cos(2πn/(N-1)); endpoints = 0.08, symmetric.
        XCTAssertEqual(w[0], 0.08, accuracy: 1e-5)
        XCTAssertEqual(w[399], 0.08, accuracy: 1e-5)
        XCTAssertGreaterThan(w[200], 0.99)  // near-unity peak in the middle
    }

    func testMelFilterbank() {
        let mel = KaldiFbank.melFilterbank(
            numBins: 80, numFftBins: 256, fftSize: 512,
            sampleRate: 16_000, lowFreq: 20, highFreq: 8_000)
        XCTAssertEqual(mel.count, 80)
        XCTAssertEqual(mel[0].count, 256)
        // Triangular filters are non-negative and march up in frequency.
        for row in mel { for v in row { XCTAssertGreaterThanOrEqual(v, 0) } }
        func peakBin(_ r: [Float]) -> Int { r.indices.max(by: { r[$0] < r[$1] })! }
        XCTAssertLessThan(peakBin(mel[10]), peakBin(mel[70]))
        XCTAssertGreaterThan(mel[40].reduce(0, +), 0)  // non-empty filter
    }
}
