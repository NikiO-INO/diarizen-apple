import AVFoundation
import Foundation

/// Decode an audio file to mono Float samples at a target sample rate (16 kHz).
/// Uses AVAudioFile + a converter so we depend on no external tools at runtime.
public enum AudioLoader {

    public static func loadMono(url: URL, sampleRate: Double = 16_000) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let outFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        )!
        guard let converter = AVAudioConverter(from: file.processingFormat, to: outFormat) else {
            throw DiariZenError.inference("cannot create audio converter")
        }

        var out = [Float]()
        let inBuf = AVAudioPCMBuffer(
            pcmFormat: file.processingFormat,
            frameCapacity: 1 << 16
        )!

        while true {
            try file.read(into: inBuf)
            if inBuf.frameLength == 0 { break }
            let ratio = sampleRate / file.processingFormat.sampleRate
            let cap = AVAudioFrameCount(Double(inBuf.frameLength) * ratio) + 1024
            let outBuf = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: cap)!
            var err: NSError?
            var fed = false
            converter.convert(to: outBuf, error: &err) { _, status in
                if fed { status.pointee = .noDataNow; return nil }
                fed = true
                status.pointee = .haveData
                return inBuf
            }
            if let err { throw DiariZenError.inference("resample: \(err)") }
            if let ch = outBuf.floatChannelData, outBuf.frameLength > 0 {
                out.append(contentsOf: UnsafeBufferPointer(start: ch[0], count: Int(outBuf.frameLength)))
            }
        }
        return out
    }
}
