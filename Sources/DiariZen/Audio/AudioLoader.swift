import AVFoundation
import Foundation

/// Decode an audio file to mono Float samples at a target sample rate (16 kHz).
/// Uses AVAudioFile so we depend on no external tools at runtime. When the input
/// is already mono at the target rate (the common case) the samples are returned
/// directly; otherwise an AVAudioConverter resamples/downmixes.
public enum AudioLoader {

    public static func loadMono(url: URL, sampleRate: Double = 16_000) throws -> [Float] {
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw DiariZenError.inference("open \(url.lastPathComponent): \(error.localizedDescription)")
        }
        let inFormat = file.processingFormat  // always non-interleaved float32

        guard let inBuf = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: AVAudioFrameCount(file.length)),
              file.length > 0 else {
            return []
        }
        do {
            try file.read(into: inBuf)
        } catch {
            throw DiariZenError.inference("read \(url.lastPathComponent): \(error.localizedDescription)")
        }

        // Fast path: already mono at the target rate — return the channel directly.
        if inFormat.sampleRate == sampleRate, inFormat.channelCount == 1,
           let ch = inBuf.floatChannelData {
            return Array(UnsafeBufferPointer(start: ch[0], count: Int(inBuf.frameLength)))
        }

        // Otherwise resample / downmix to mono float32 at `sampleRate`.
        guard let outFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false
        ), let converter = AVAudioConverter(from: inFormat, to: outFormat) else {
            throw DiariZenError.inference("cannot create audio converter")
        }

        let ratio = sampleRate / inFormat.sampleRate
        let outCapacity = AVAudioFrameCount(Double(inBuf.frameLength) * ratio) + 4096
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: outCapacity) else {
            throw DiariZenError.inference("cannot allocate output buffer")
        }

        var err: NSError?
        var fed = false
        let status = converter.convert(to: outBuf, error: &err) { _, inputStatus in
            if fed { inputStatus.pointee = .endOfStream; return nil }
            fed = true
            inputStatus.pointee = .haveData
            return inBuf
        }
        if status == .error {
            throw DiariZenError.inference("resample: \(err?.localizedDescription ?? "unknown")")
        }
        guard let ch = outBuf.floatChannelData else {
            throw DiariZenError.inference("no output channel data")
        }
        return Array(UnsafeBufferPointer(start: ch[0], count: Int(outBuf.frameLength)))
    }
}
