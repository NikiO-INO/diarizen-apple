import Foundation

/// RTTM hypothesis I/O. The format hosts (incl. meetlify's `turns_from_rttm`)
/// already parse: `SPEAKER <file> 1 <start> <dur> <NA> <NA> speaker_<n> <NA> <NA>`.
public enum RTTM {
    public static func serialize(_ turns: [SpeakerTurn], fileId: String) -> String {
        turns
            .sorted { $0.start < $1.start }
            .map { t in
                let dur = max(0, t.end - t.start)
                return String(
                    format: "SPEAKER %@ 1 %.3f %.3f <NA> <NA> speaker_%d <NA> <NA>",
                    fileId, t.start, dur, t.speaker
                )
            }
            .joined(separator: "\n")
    }
}
