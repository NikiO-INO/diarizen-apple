import DiariZen
import Foundation

// Thin CLI: audio → RTTM, mirroring the `fluidaudiocli nemotron3-diarize`
// interface so hosts (meetlify) can drop it in as a sidecar diarization engine.
//
//   diarizen-cli <audio.wav> --models <dir> [--compute-units all|cpu-ane|cpu]
//                            [--output <rttm>]

func usage() -> Never {
    FileHandle.standardError.write(Data("""
        DiariZen (Apple Silicon)
          diarizen-cli <audio> [options]

        Options:
          --models <dir>          Directory with Segmentation.mlmodelc + Embedding.mlmodelc
          --compute-units <u>     all | cpu-ane | cpu   (default: all)
          --output <file>         Write RTTM (default: stdout)

        """.utf8))
    exit(2)
}

var audioPath: String?
var modelsDir: String?
var outputPath: String?
var policy: CoreMLBackend.ComputePolicy = .all

var args = Array(CommandLine.arguments.dropFirst())
var i = 0
while i < args.count {
    let a = args[i]
    switch a {
    case "--models": i += 1; modelsDir = args[safe: i]
    case "--output": i += 1; outputPath = args[safe: i]
    case "--compute-units":
        i += 1
        switch args[safe: i] {
        case "cpu": policy = .cpuOnly
        case "cpu-ane": policy = .cpuAndNeuralEngine
        default: policy = .all
        }
    case "--help", "-h": usage()
    default:
        if !a.hasPrefix("--"), audioPath == nil { audioPath = a }
    }
    i += 1
}

guard let audioPath, let modelsDir else { usage() }

do {
    let waveform = try AudioLoader.loadMono(url: URL(fileURLWithPath: audioPath))
    let segURL = URL(fileURLWithPath: modelsDir).appendingPathComponent("Segmentation.mlmodelc")
    let embURL = URL(fileURLWithPath: modelsDir).appendingPathComponent("Embedding.mlmodelc")
    let pipeline = DiarizationPipeline(
        segmentation: CoreMLSegmentation(backend: try CoreMLBackend(compiledModelURL: segURL, policy: policy)),
        embeddings: CoreMLEmbedding(backend: try CoreMLBackend(compiledModelURL: embURL, policy: policy))
    )
    let turns = try pipeline.diarize(waveform: waveform, sampleRate: 16_000)
    let fileId = URL(fileURLWithPath: audioPath).deletingPathExtension().lastPathComponent
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
