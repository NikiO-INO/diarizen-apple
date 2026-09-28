import XCTest
@testable import DiariZen

final class HungarianTests: XCTestCase {
    func testMaxAssignmentSquare() {
        // Best is row0→col1 (5) + row1→col0 (4) = 9.
        let pairs = Hungarian.maxAssignment([[1, 5], [4, 2]]).sorted { $0.0 < $1.0 }
        XCTAssertEqual(pairs.map { $0.1 }, [1, 0])
    }

    func testMaxAssignmentRectangularMoreRows() {
        // 3 speakers, 2 clusters → 2 pairs, picking the two best rows.
        let cost = [[0.1, 0.9], [0.8, 0.2], [0.05, 0.05]]
        let pairs = Hungarian.maxAssignment(cost)
        XCTAssertEqual(pairs.count, 2)
        // col0 → row1 (0.8), col1 → row0 (0.9)
        let byCol = Dictionary(uniqueKeysWithValues: pairs.map { ($0.1, $0.0) })
        XCTAssertEqual(byCol[0], 1)
        XCTAssertEqual(byCol[1], 0)
    }
}

final class AHCTests: XCTestCase {
    func testCentroidLinkageAndCut() {
        // Two near points + one far → cut at 0.6 gives {0,1} and {2}.
        let pts: [[Double]] = [[0, 0], [0, 0.1], [10, 10]]
        let merges = AHC.centroidLinkage(pts)
        XCTAssertEqual(merges.count, 2)
        let labels = AHC.fclusterDistance(merges, n: 3, threshold: 0.6)
        XCTAssertEqual(labels[0], labels[1])
        XCTAssertNotEqual(labels[0], labels[2])
    }
}

final class ReconstructionTests: XCTestCase {
    func testCosineDistance() {
        XCTAssertEqual(cosineDistance([1, 0], [1, 0]), 0, accuracy: 1e-12)
        XCTAssertEqual(cosineDistance([1, 0], [0, 1]), 1, accuracy: 1e-12)
    }

    func testBinarizeExtractsRuns() {
        // frame ts(i) = i; one cluster active on frames 0,1 then 3.
        let frame = FrameResolution(start: 0, duration: 0, step: 1)
        let scores: [[Double]] = [[1], [1], [0], [1]]
        let turns = Reconstruction.binarize(scores, frame: frame).sorted { $0.start < $1.start }
        XCTAssertEqual(turns.count, 2)
        XCTAssertEqual(turns[0].start, 0, accuracy: 1e-9)
        XCTAssertEqual(turns[0].end, 2, accuracy: 1e-9)     // ends at first inactive frame
        XCTAssertEqual(turns[1].start, 3, accuracy: 1e-9)
        XCTAssertEqual(turns[1].end, 3, accuracy: 1e-9)     // active at end
    }
}
