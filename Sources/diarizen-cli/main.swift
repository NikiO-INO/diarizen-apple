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
var dumpFeaturesPath: String?
var debugClusterPath: String?
var benchmarkRuns: Int?
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
    case "--dump-features": i += 1; dumpFeaturesPath = args[safe: i]
    case "--debug-cluster": i += 1; debugClusterPath = args[safe: i]
    case "--benchmark": i += 1; benchmarkRuns = args[safe: i].flatMap { Int($0) } ?? 5
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

if case .all = policy {
    let warn = "warning: --compute-units all (CPU+GPU+ANE) deadlocks the segmentation "
        + "predict on this model; use cpu-gpu (default) or cpu-ane.\n"
    FileHandle.standardError.write(Data(warn.utf8))
}

// Phase 3 parity mode: PLDA transform + VBx on injected embeddings (no audio/model).
if let debugClusterPath {
    struct In: Decodable { let train_emb: [[Float]]; let ahc: [Int]; let Fa: Double; let Fb: Double; let maxIters: Int }
    struct Out: Encodable { let fea: [[Double]]; let gamma: [[Double]]; let pi: [Double] }
    guard let modelsDir else { usage() }
    do {
        let input = try JSONDecoder().decode(In.self, from: Data(contentsOf: URL(fileURLWithPath: debugClusterPath)))
        let plda = try PLDA(contentsOf: URL(fileURLWithPath: modelsDir).appendingPathComponent("plda_transform.json"))
        let fea = plda.transform(input.train_emb)
        let (gamma, pi) = VBx.cluster(ahcInit: input.ahc, fea: fea, phi: plda.phi,
                                      Fa: input.Fa, Fb: input.Fb, maxIters: input.maxIters)
        let outURL = URL(fileURLWithPath: outputPath ?? "cluster_debug.json")
        try JSONEncoder().encode(Out(fea: fea, gamma: gamma, pi: pi)).write(to: outURL)
        FileHandle.standardError.write(Data("wrote cluster debug → \(outURL.path)\n".utf8))
        exit(0)
    } catch {
        FileHandle.standardError.write(Data("error: \(error)\n".utf8)); exit(1)
    }
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
    let plda = try PLDA(contentsOf: URL(fileURLWithPath: modelsDir).appendingPathComponent("plda_transform.json"))
    let pipeline = DiarizationPipeline(
        segmentation: segmentation,
        embeddings: CoreMLEmbedding(backend: try CoreMLBackend(modelURL: embURL, policy: policy)),
        clustering: VBxClustering(plda: plda)
    )

    // Benchmark mode: run the full pipeline N times, report per-stage medians + RTF.
    if let benchmarkRuns {
        let audioDur = Double(waveform.count) / 16_000.0
        _ = try pipeline.diarize(waveform: waveform, sampleRate: 16_000)  // warm-up
        var seg = [Double](), emb = [Double](), clu = [Double](), rec = [Double](), tot = [Double]()
        for _ in 0..<benchmarkRuns {
            _ = try pipeline.diarize(waveform: waveform, sampleRate: 16_000)
            let t = pipeline.timings
            seg.append(t.segmentation); emb.append(t.embedding)
            clu.append(t.clustering); rec.append(t.reconstruction); tot.append(t.total)
        }
        func med(_ a: [Double]) -> Double { let s = a.sorted(); return s[s.count / 2] }
        func f(_ x: Double) -> String { String(format: "%.3f", x) }
        let mt = med(tot)
        print("""
        backend: native CoreML (\(policy)) + Swift; runs=\(benchmarkRuns); audio=\(f(audioDur))s
          segmentation   \(f(med(seg)))s
          embedding      \(f(med(emb)))s
          clustering     \(f(med(clu)))s
          reconstruction \(f(med(rec)))s
          total (median) \(f(mt))s
          RTF            \(String(format: "%.4f", mt / audioDur))
        """)
        exit(0)
    }

    // Phase 2 parity mode: dump the pre-clustering features (binarized + embeddings).
    if let dumpFeaturesPath {
        let feats = try pipeline.extractFeatures(waveform: waveform, sampleRate: 16_000)
        struct Features: Encodable { let binarized: [[[Float]]]; let embeddings: [[[Float]]] }
        let data = try JSONEncoder().encode(Features(binarized: feats.binarized, embeddings: feats.embeddings))
        try data.write(to: URL(fileURLWithPath: dumpFeaturesPath))
        let bShape = "\(feats.binarized.count)×\(feats.binarized.first?.count ?? 0)×\(feats.binarized.first?.first?.count ?? 0)"
        let eShape = "\(feats.embeddings.count)×\(feats.embeddings.first?.count ?? 0)"
        let msg = "dumped features: binarized \(bShape), embeddings \(eShape) → \(dumpFeaturesPath)\n"
        FileHandle.standardError.write(Data(msg.utf8))
        exit(0)
    }

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
