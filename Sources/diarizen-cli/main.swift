import CoreML
import DiariZen
import Foundation

// Thin CLI: audio → RTTM, mirroring the `fluidaudiocli nemotron3-diarize`
// interface so hosts (meetlify) can drop it in as a sidecar diarization engine.
//
//   diarizen-cli <audio.wav> --models <dir> [--compute-units cpu-gpu|all|cpu-ane|cpu]
//                            [--output <rttm>]
//                            [--dump-segmentation <json>]  # Phase 2 parity dump
//
// --compute-units default is cpu-gpu: the ANE path (all / cpu-ane) DEADLOCKS the
// first segmentation predict on this model (see CoreMLBackend.ComputePolicy).

func usage() -> Never {
    FileHandle.standardError.write(Data("""
        DiariZen (Apple Silicon)
          diarizen-cli <audio> [options]

        Options:
          --models <dir>            Directory with Segmentation.mlpackage (+ Embedding.mlpackage)
          --compute-units <u>       cpu-gpu | all | cpu-ane | cpu   (default: cpu-gpu)
          --output <file>           Write RTTM (default: stdout)
          --dump-segmentation <f>   Run segmentation on the first 16 s window, write
                                    the [frames][classes] log-probs as JSON, and exit
                                    (Phase 2 parity against the Python backend)
          --dump-fbank <f>          Compute the Kaldi fbank of the first 16 s window,
                                    write [frames][mels] as JSON, and exit (no model
                                    needed; parity against torchaudio kaldi.fbank)
          --raw-input <f>           Read the window as little-endian Float32 samples
                                    from <f> instead of decoding <audio> (isolates
                                    inference from the audio loader for parity)

        """.utf8))
    exit(2)
}

/// Read a little-endian Float32 sample file into `[Float]`.
func readRawFloat32(_ path: String) throws -> [Float] {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    return data.withUnsafeBytes { raw in Array(raw.bindMemory(to: Float32.self)) }
}

/// Prefer an `.mlpackage` (what export_coreml.py writes); fall back to a compiled
/// `.mlmodelc`. Returns nil if neither exists.
func resolveModel(dir: String, base: String) -> URL? {
    let d = URL(fileURLWithPath: dir)
    for ext in ["mlpackage", "mlmodelc"] {
        let u = d.appendingPathComponent("\(base).\(ext)")
        if FileManager.default.fileExists(atPath: u.path) { return u }
    }
    return nil
}

var audioPath: String?
var modelsDir: String?
var outputPath: String?
var dumpSegPath: String?
var dumpFbankPath: String?
var dumpEmbPath: String?
var weightsRawPath: String?
var rawInputPath: String?
var policy: CoreMLBackend.ComputePolicy = .cpuAndGPU

var args = Array(CommandLine.arguments.dropFirst())
var i = 0
while i < args.count {
    let a = args[i]
    switch a {
    case "--models": i += 1; modelsDir = args[safe: i]
    case "--output": i += 1; outputPath = args[safe: i]
    case "--dump-segmentation": i += 1; dumpSegPath = args[safe: i]
    case "--dump-fbank": i += 1; dumpFbankPath = args[safe: i]
    case "--dump-embedding": i += 1; dumpEmbPath = args[safe: i]
    case "--weights-raw": i += 1; weightsRawPath = args[safe: i]
    case "--raw-input": i += 1; rawInputPath = args[safe: i]
    case "--compute-units":
        i += 1
        switch args[safe: i] {
        case "cpu": policy = .cpuOnly
        case "cpu-ane": policy = .cpuAndNeuralEngine
        case "all": policy = .all
        default: policy = .cpuAndGPU
        }
    case "--help", "-h": usage()
    default:
        if !a.hasPrefix("--"), audioPath == nil { audioPath = a }
    }
    i += 1
}

guard audioPath != nil || rawInputPath != nil else { usage() }

do {
    let waveform: [Float]
    if let rawInputPath {
        waveform = try readRawFloat32(rawInputPath)
    } else {
        waveform = try AudioLoader.loadMono(url: URL(fileURLWithPath: audioPath!))
    }

    // Fbank parity dump needs no model.
    if let dumpFbankPath {
        let window = Array(waveform.prefix(CoreMLSegmentation.windowSamples))
        let fbank = KaldiFbank().compute(window)
        try JSONEncoder().encode(fbank).write(to: URL(fileURLWithPath: dumpFbankPath))
        FileHandle.standardError.write(Data(
            "dumped fbank \(fbank.count)×\(fbank.first?.count ?? 0) → \(dumpFbankPath)\n".utf8))
        exit(0)
    }

    guard let modelsDir else { usage() }

    // Embedding parity dump: Swift fbank → CoreML ResNet on the first 16 s window.
    if let dumpEmbPath {
        guard let embURL = resolveModel(dir: modelsDir, base: "Embedding") else {
            throw DiariZenError.modelLoad("Embedding.mlpackage/.mlmodelc not found in \(modelsDir)")
        }
        let embedder = CoreMLEmbedding(backend: try CoreMLBackend(modelURL: embURL, policy: policy))
        let window = Array(waveform.prefix(CoreMLEmbedding.cropSamples))
        let weights = try weightsRawPath.map(readRawFloat32)
        let vec = try embedder.embed(region: window, weights: weights)
        try JSONEncoder().encode(vec).write(to: URL(fileURLWithPath: dumpEmbPath))
        FileHandle.standardError.write(Data("dumped embedding dim=\(vec.count) → \(dumpEmbPath)\n".utf8))
        exit(0)
    }

    guard let segURL = resolveModel(dir: modelsDir, base: "Segmentation") else {
        throw DiariZenError.modelLoad("Segmentation.mlpackage/.mlmodelc not found in \(modelsDir)")
    }
    let segmentation = CoreMLSegmentation(
        backend: try CoreMLBackend(modelURL: segURL, policy: policy)
    )

    // Phase 2 parity mode: dump the segmentation of the first window and exit.
    if let dumpSegPath {
        let n = CoreMLSegmentation.windowSamples
        let window = Array(waveform.prefix(n))
        let seg = try segmentation.segment(chunk: window)
        let data = try JSONEncoder().encode(seg.frames)
        try data.write(to: URL(fileURLWithPath: dumpSegPath))
        FileHandle.standardError.write(Data(
            "dumped segmentation \(seg.frames.count)×\(seg.frames.first?.count ?? 0) → \(dumpSegPath)\n".utf8))
        exit(0)
    }

    guard let embURL = resolveModel(dir: modelsDir, base: "Embedding") else {
        throw DiariZenError.modelLoad("Embedding.mlpackage/.mlmodelc not found in \(modelsDir)")
    }
    let pipeline = DiarizationPipeline(
        segmentation: segmentation,
        embeddings: CoreMLEmbedding(backend: try CoreMLBackend(modelURL: embURL, policy: policy))
    )
    let turns = try pipeline.diarize(waveform: waveform, sampleRate: 16_000)
    let fileId = audioPath.map { URL(fileURLWithPath: $0).deletingPathExtension().lastPathComponent } ?? "audio"
    let rttm = RTTM.serialize(turns, fileId: fileId)
    if let outputPath {
        try rttm.write(toFile: outputPath, atomically: true, encoding: .utf8)
    } else {
        print(rttm)
    }
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}

extension Array {
    subscript(safe idx: Int) -> Element? { indices.contains(idx) ? self[idx] : nil }
}
