//
//  HNSWIndexTests.swift
//  SemanticNotesTests
//

import Foundation
import Testing

@testable import SemanticNotes

/// テスト用の決定的な乱数(シード固定で毎回同じベクトル列を作る)
private struct TestRNG: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}

private func randomUnitVector(_ dimension: Int, using rng: inout TestRNG) -> [Float] {
    var vector = (0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) }
    let norm = vector.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
    if norm > 0 { vector = vector.map { $0 / norm } }
    return vector
}

private func makeVectors(count: Int, dimension: Int, seed: UInt64) -> [[Float]] {
    var rng = TestRNG(state: seed)
    return (0..<count).map { _ in randomUnitVector(dimension, using: &rng) }
}

/// recall@k: 総当たり(真値)の上位 k 件のうち、近似検索が見つけられた割合
private func recall(approx: [VectorSearchResult], truth: [VectorSearchResult]) -> Double {
    guard !truth.isEmpty else { return 1 }
    let truthIDs = Set(truth.map(\.id))
    let found = approx.map(\.id).filter { truthIDs.contains($0) }.count
    return Double(found) / Double(truthIDs.count)
}

// MARK: - BinaryHeap

struct BinaryHeapTests {
    @Test func 比較関数に従って順に取り出せる() {
        var heap = BinaryHeap<Int> { $0 < $1 }
        for value in [5, 1, 4, 2, 8, 0, 7] {
            heap.push(value)
        }
        var popped: [Int] = []
        while let value = heap.pop() {
            popped.append(value)
        }
        #expect(popped == [0, 1, 2, 4, 5, 7, 8])
        #expect(heap.isEmpty)
    }

    @Test func ランダム入力でもソート結果と一致する() {
        var rng = TestRNG(state: 7)
        let values = (0..<500).map { _ in Int.random(in: 0..<10_000, using: &rng) }
        var heap = BinaryHeap<Int> { $0 > $1 } // 最大ヒープ
        values.forEach { heap.push($0) }
        var popped: [Int] = []
        while let value = heap.pop() {
            popped.append(value)
        }
        #expect(popped == values.sorted(by: >))
    }
}

// MARK: - HNSWIndex

struct HNSWIndexTests {
    private func configuration(
        seed: UInt64 = 42, m: Int = 16, efConstruction: Int = 100, efSearch: Int = 64
    ) -> HNSWIndex.Configuration {
        HNSWIndex.Configuration(m: m, efConstruction: efConstruction, efSearch: efSearch, seed: seed)
    }

    @Test func 空インデックスは空を返す() async {
        let index = HNSWIndex(dimension: 4, configuration: configuration())
        #expect(await index.search([1, 0, 0, 0], k: 10).isEmpty)
        #expect(await index.count == 0)
    }

    @Test func 少数ノードでは総当たりと完全一致する() async {
        // efSearch をノード数以上にすれば実質的に全探索になり、結果は真値と一致するはず
        let vectors = makeVectors(count: 200, dimension: 16, seed: 1)
        let hnsw = HNSWIndex(dimension: 16, configuration: configuration(efSearch: 256))
        let brute = BruteForceIndex(dimension: 16)
        var ids: [UUID] = []
        for vector in vectors {
            let id = UUID()
            ids.append(id)
            await hnsw.upsert(id: id, vector: vector)
            await brute.upsert(id: id, vector: vector)
        }

        var queryRNG = TestRNG(state: 99)
        for _ in 0..<10 {
            let query = randomUnitVector(16, using: &queryRNG)
            let approx = await hnsw.search(query, k: 10)
            let truth = await brute.search(query, k: 10)
            #expect(recall(approx: approx, truth: truth) == 1.0)
        }
    }

    @Test func シード固定のデータでrecallが基準を満たす() async {
        let vectors = makeVectors(count: 1_000, dimension: 64, seed: 2)
        let hnsw = HNSWIndex(dimension: 64, configuration: configuration())
        let brute = BruteForceIndex(dimension: 64)
        for vector in vectors {
            let id = UUID()
            await hnsw.upsert(id: id, vector: vector)
            await brute.upsert(id: id, vector: vector)
        }

        var queryRNG = TestRNG(state: 100)
        var total = 0.0
        let queryCount = 20
        for _ in 0..<queryCount {
            let query = randomUnitVector(64, using: &queryRNG)
            let approx = await hnsw.search(query, k: 10)
            let truth = await brute.search(query, k: 10)
            total += recall(approx: approx, truth: truth)
        }
        let average = total / Double(queryCount)
        print("[recall] N=1000, d=64, M=16, efC=100, efS=64: recall@10 = \(average)")
        #expect(average >= 0.95)
    }

    @Test func upsertは同じIDのベクトルを置き換える() async {
        let index = HNSWIndex(dimension: 4, configuration: configuration())
        let target = UUID()
        await index.upsert(id: target, vector: [1, 0, 0, 0])
        await index.upsert(id: UUID(), vector: [0, 1, 0, 0])
        await index.upsert(id: UUID(), vector: [0, 0, 1, 0])

        await index.upsert(id: target, vector: [0, 0, 0, 1]) // 置き換え

        #expect(await index.count == 3)
        let newLocation = await index.search([0, 0, 0, 1], k: 1)
        #expect(newLocation.first?.id == target)
        // 旧ベクトルの位置では他のノードより遠くなっている(墓標が結果に出ない)
        let oldLocation = await index.search([1, 0, 0, 0], k: 3)
        #expect(oldLocation.first { $0.id == target }.map { $0.score < 0.5 } ?? true)
    }

    @Test func removeしたノードは検索に出ない() async {
        let index = HNSWIndex(dimension: 4, configuration: configuration())
        let doomed = UUID()
        await index.upsert(id: doomed, vector: [1, 0, 0, 0])
        await index.upsert(id: UUID(), vector: [0.9, 0.43, 0, 0])
        await index.upsert(id: UUID(), vector: [0, 1, 0, 0])

        await index.remove(id: doomed)

        #expect(await index.count == 2)
        let results = await index.search([1, 0, 0, 0], k: 3)
        #expect(results.count == 2)
        #expect(!results.map(\.id).contains(doomed))
    }

    @Test func 大量削除で再構築され墓標が回収される() async {
        let vectors = makeVectors(count: 300, dimension: 8, seed: 3)
        let index = HNSWIndex(dimension: 8, configuration: configuration())
        var ids: [UUID] = []
        for vector in vectors {
            let id = UUID()
            ids.append(id)
            await index.upsert(id: id, vector: vector)
        }

        for id in ids.prefix(150) {
            await index.remove(id: id)
        }

        let stats = await index.stats()
        #expect(stats.liveCount == 150)
        // 途中(151件目の削除)で再構築が走るので、150件分の墓標がそのまま残ることはない。
        // その後の削除で溜まり直した分は、しきい値(64 かつ生存数の半分)以内に収まる
        #expect(stats.tombstoneCount < 150, "一度も再構築されていない")
        #expect(stats.tombstoneCount <= 64 || stats.tombstoneCount * 2 <= stats.liveCount)
        // 再構築後も検索が機能する
        var queryRNG = TestRNG(state: 5)
        let results = await index.search(randomUnitVector(8, using: &queryRNG), k: 10)
        #expect(results.count == 10)
        #expect(results.allSatisfy { !ids.prefix(150).contains($0.id) })
    }

    @Test func 保存と読み込みで検索結果が変わらない() async throws {
        let vectors = makeVectors(count: 300, dimension: 16, seed: 4)
        let original = HNSWIndex(dimension: 16, configuration: configuration())
        for vector in vectors {
            await original.upsert(id: UUID(), vector: vector)
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("hnsw-test-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try await original.save(to: url)

        let loaded = try HNSWIndex(contentsOf: url)
        #expect(await loaded.count == 300)
        #expect(loaded.dimension == 16)

        var queryRNG = TestRNG(state: 6)
        for _ in 0..<5 {
            let query = randomUnitVector(16, using: &queryRNG)
            let before = await original.search(query, k: 10)
            let after = await loaded.search(query, k: 10)
            #expect(before.map(\.id) == after.map(\.id))
            #expect(zip(before, after).allSatisfy { abs($0.score - $1.score) < 1e-5 })
        }
    }

    @Test func 壊れたファイルはエラーになる() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("hnsw-corrupt-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }

        try Data([0x00, 0x01, 0x02]).write(to: url) // magic にすら足りない
        #expect(throws: (any Error).self) {
            _ = try HNSWIndex(contentsOf: url)
        }

        // 正しい magic + でたらめな中身(途中で切れている)
        var data = Data()
        data.appendForTest(0x484E_5357)
        data.appendForTest(1)
        data.appendForTest(384)
        try data.write(to: url)
        #expect(throws: (any Error).self) {
            _ = try HNSWIndex(contentsOf: url)
        }
    }
}

private extension Data {
    mutating func appendForTest(_ value: UInt32) {
        Swift.withUnsafeBytes(of: value.littleEndian) { append(contentsOf: $0) }
    }
}
