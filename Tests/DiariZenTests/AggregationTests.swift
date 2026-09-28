import XCTest
@testable import DiariZen

final class PowersetTests: XCTestCase {
    func testMappingMatchesPyannoteOrder() {
        let ps = Powerset(numSpeakers: 4, maxSetSize: 2)
        XCTAssertEqual(ps.numPowersetClasses, 11)
        XCTAssertEqual(ps.mapping[0], [0, 0, 0, 0])   // silence
        XCTAssertEqual(ps.mapping[1], [1, 0, 0, 0])   // singles
        XCTAssertEqual(ps.mapping[4], [0, 0, 0, 1])
        XCTAssertEqual(ps.mapping[5], [1, 1, 0, 0])   // first pair (0,1)
        XCTAssertEqual(ps.mapping[10], [0, 0, 1, 1])  // last pair (2,3)
    }

    func testCombinations() {
        XCTAssertEqual(Powerset.combinations(4, 0), [[]])
        XCTAssertEqual(Powerset.combinations(4, 2), [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    }

    func testToMultilabelArgmax() {
        let ps = Powerset(numSpeakers: 4, maxSetSize: 2)
        // frame 0 → class 5 (pair 0,1); frame 1 → class 4 (speaker 3)
        var f0 = [Float](repeating: -10, count: 11); f0[5] = 2.0
        var f1 = [Float](repeating: -10, count: 11); f1[4] = 1.0
        let out = ps.toMultilabel([f0, f1])
        XCTAssertEqual(out[0], [1, 1, 0, 0])
        XCTAssertEqual(out[1], [0, 0, 0, 1])
    }
}

final class MedianFilterTests: XCTestCase {
    func testSpikeRemovedWindow3() {
        var frames: [[Float]] = [[0], [1], [0]]  // one-frame spike
        DiarizationPipeline.medianFilterFrames(&frames, window: 3)
        XCTAssertEqual(frames, [[0], [0], [0]])
    }

    func testReflectIndexHalfSampleSymmetric() {
        XCTAssertEqual(DiarizationPipeline.reflectIndex(-1, 5), 0)
        XCTAssertEqual(DiarizationPipeline.reflectIndex(-2, 5), 1)
        XCTAssertEqual(DiarizationPipeline.reflectIndex(5, 5), 4)
        XCTAssertEqual(DiarizationPipeline.reflectIndex(6, 5), 3)
        XCTAssertEqual(DiarizationPipeline.reflectIndex(2, 5), 2)  // in-range unchanged
    }
}
