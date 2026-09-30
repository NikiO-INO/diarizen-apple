import Foundation
import XCTest

@testable import DiariZen

final class CoreMLBackendTests: XCTestCase {

    /// The `.mlmodelc` cache is reused only when it exists and is at least as new
    /// as the source `.mlpackage`; re-exporting the model must invalidate it.
    func testCacheIsFreshRespectsModificationTimes() throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }

        let source = dir.appendingPathComponent("model.mlpackage")
        let cache = dir.appendingPathComponent("model.mlmodelc")

        // No cache yet -> not fresh.
        try "src".write(to: source, atomically: true, encoding: .utf8)
        XCTAssertFalse(CoreMLBackend.cacheIsFresh(cacheURL: cache, sourceURL: source))

        // Cache newer than source -> fresh.
        try "compiled".write(to: cache, atomically: true, encoding: .utf8)
        try fm.setAttributes([.modificationDate: Date(timeIntervalSince1970: 2000)], ofItemAtPath: source.path)
        try fm.setAttributes([.modificationDate: Date(timeIntervalSince1970: 3000)], ofItemAtPath: cache.path)
        XCTAssertTrue(CoreMLBackend.cacheIsFresh(cacheURL: cache, sourceURL: source))

        // Source re-exported after the cache -> stale.
        try fm.setAttributes([.modificationDate: Date(timeIntervalSince1970: 4000)], ofItemAtPath: source.path)
        XCTAssertFalse(CoreMLBackend.cacheIsFresh(cacheURL: cache, sourceURL: source))
    }
}
