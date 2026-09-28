import CoreML
import XCTest
@testable import DiariZen

final class SegmentationDecodeTests: XCTestCase {
    /// (1, frames, classes) C-contiguous MLMultiArray → [frames][classes], float32 path.
    func testToFramesReshapesContiguousFloat32() throws {
        let arr = try MLMultiArray(shape: [1, 2, 3], dataType: .float32)
        for i in 0..<6 { arr[i] = NSNumber(value: Float(i)) }  // 0..5 in C order
        let frames = CoreMLSegmentation.toFrames(arr)
        XCTAssertEqual(frames.count, 2)
        XCTAssertEqual(frames[0], [0, 1, 2])
        XCTAssertEqual(frames[1], [3, 4, 5])
    }

    /// The dtype-agnostic branch (float16 output) must decode identically.
    func testToFramesFloat16Path() throws {
        let arr = try MLMultiArray(shape: [1, 2, 3], dataType: .float16)
        for i in 0..<6 { arr[i] = NSNumber(value: Float(i)) }
        let frames = CoreMLSegmentation.toFrames(arr)
        XCTAssertEqual(frames[0], [0, 1, 2])
        XCTAssertEqual(frames[1], [3, 4, 5])
    }
}
