import XCTest
@testable import DiariZen

final class RTTMTests: XCTestCase {
    func testSerializeSortsAndFormats() {
        let turns = [
            SpeakerTurn(start: 3.0, end: 4.5, speaker: 2),
            SpeakerTurn(start: 0.5, end: 2.5, speaker: 1),
        ]
        let rttm = RTTM.serialize(turns, fileId: "clip")
        let lines = rttm.split(separator: "\n").map(String.init)
        XCTAssertEqual(lines.count, 2)
        // Sorted by start; RTTM start+duration; speaker_<n>.
        XCTAssertEqual(lines[0], "SPEAKER clip 1 0.500 2.000 <NA> <NA> speaker_1 <NA> <NA>")
        XCTAssertEqual(lines[1], "SPEAKER clip 1 3.000 1.500 <NA> <NA> speaker_2 <NA> <NA>")
    }

    func testEmptyIsEmptyString() {
        XCTAssertEqual(RTTM.serialize([], fileId: "x"), "")
    }
}
