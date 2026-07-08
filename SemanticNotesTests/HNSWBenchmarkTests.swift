//
//  HNSWBenchmarkTests.swift
//  SemanticNotesTests
//
//  PLAN Phase 5 の計測: 総当たりとの recall@10・速度・メモリ比較と、
//  パラメータ(M, efConstruction, efSearch)の影響調査。
//  実行に数分かかるため、環境変数で明示したときだけ動かす:
//    TEST_RUNNER_RUN_BENCHMARKS=1 xcodebuild ... test -only-testing:SemanticNotesTests/HNSWBenchmarks
//
//  データ分布について: 一様ランダムな高次元ベクトルは全ペアがほぼ等距離になり
//  (次元の呪い)、グラフ系 ANN には最悪ケースになる。実際の文埋め込みは
//  話題ごとのクラスタ構造を持つため、代表ケースにはクラスタ混合データを使い、
//  一様ランダムは限界を知るストレスケースとして別テストで扱う。
//

import Foundation
import Testing

@testable import SemanticNotes

private nonisolated(unsafe) let runBenchmarks =
    ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"

private struct BenchRNG: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}

private func normalized(_ vector: [Float]) -> [Float] {
    let norm = vector.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
    guard norm > 0 else { return vector }
    return vector.map { $0 / norm }
}

private func uniformUnitVectors(count: Int, dimension: Int, seed: UInt64) -> [[Float]] {
    var rng = BenchRNG(state: seed)
    return (0..<count).map { _ in
        normalized((0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) })
    }
}

private func gaussian(using rng: inout BenchRNG) -> Float {
    // Box-Muller 変換。一様乱数2つから標準正規乱数を作る
    let u1 = max(Double(rng.next() >> 11) * 0x1.0p-53, .leastNonzeroMagnitude)
    let u2 = Double(rng.next() >> 11) * 0x1.0p-53
    return Float((-2 * log(u1)).squareRoot() * cos(2 * .pi * u2))
}

/// 文埋め込みを模したクラスタ混合データ(中心 + ガウスノイズ、正規化済み)。
/// データ点とクエリを同じ中心群から生成する(検索は「どこかの話題の近く」で起きる)。
private func clusteredDataset(
    dataCount: Int, queryCount: Int, dimension: Int, clusters: Int, seed: UInt64
) -> (data: [[Float]], queries: [[Float]]) {
    var rng = BenchRNG(state: seed)
    let centers = (0..<clusters).map { _ in
        normalized((0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) })
    }
    func sample() -> [Float] {
        let center = centers[Int(rng.next() % UInt64(clusters))]
        return normalized(center.map { $0 + 0.05 * gaussian(using: &rng) })
    }
    return ((0..<dataCount).map { _ in sample() }, (0..<queryCount).map { _ in sample() })
}

private func recall(approx: [VectorSearchResult], truth: [VectorSearchResult]) -> Double {
    let truthIDs = Set(truth.map(\.id))
    guard !truthIDs.isEmpty else { return 1 }
    return Double(approx.map(\.id).filter { truthIDs.contains($0) }.count) / Double(truthIDs.count)
}

@Suite(.enabled(if: runBenchmarks, "TEST_RUNNER_RUN_BENCHMARKS=1 を付けたときだけ実行する重い計測"))
struct HNSWBenchmarks {
    private static let dimension = 384
    private static let queryCount = 50

    /// 総当たりとの比較(PLAN の完了条件を確認する本体)。クラスタ構造データ。
    @Test func 総当たりとの比較ベンチマーク() async throws {
        let clock = ContinuousClock()
        print("[bench] N, build(s), recall@10, hnsw(ms/query), brute(ms/query), 倍率, グラフ追加メモリ(MB)")

        for n in [1_000, 5_000, 10_000] {
            let (data, queries) = clusteredDataset(
                dataCount: n, queryCount: Self.queryCount,
                dimension: Self.dimension, clusters: max(20, n / 100), seed: 11
            )

            let brute = BruteForceIndex(dimension: Self.dimension)
            let hnsw = HNSWIndex(
                dimension: Self.dimension,
                configuration: .init(m: 16, efConstruction: 200, efSearch: 64, seed: 42)
            )

            // recall は ID 集合の比較なので、両インデックスに同じ ID を使うこと
            let ids = (0..<n).map { _ in UUID() }
            let buildStart = clock.now
            for (id, vector) in zip(ids, data) {
                await hnsw.upsert(id: id, vector: vector)
            }
            let buildTime = clock.now - buildStart
            for (id, vector) in zip(ids, data) {
                await brute.upsert(id: id, vector: vector)
            }

            var truths: [[VectorSearchResult]] = []
            let bruteStart = clock.now
            for query in queries {
                truths.append(await brute.search(query, k: 10))
            }
            let bruteTime = (clock.now - bruteStart) / Self.queryCount

            let hnswStart = clock.now
            var approxResults: [[VectorSearchResult]] = []
            for query in queries {
                approxResults.append(await hnsw.search(query, k: 10))
            }
            let hnswTime = (clock.now - hnswStart) / Self.queryCount

            var totalRecall = 0.0
            for (approx, truth) in zip(approxResults, truths) {
                totalRecall += recall(approx: approx, truth: truth)
            }
            let averageRecall = totalRecall / Double(Self.queryCount)

            let stats = await hnsw.stats()
            let graphMB = Double(stats.totalLinks * 4 + stats.liveCount * 16) / 1_000_000
            let speedup = bruteTime / hnswTime
            print("[bench] N=\(n): build=\(buildTime), recall=\(averageRecall), "
                + "hnsw=\(hnswTime), brute=\(bruteTime), 倍率=x\(speedup), graph=\(graphMB)MB")

            #expect(averageRecall >= 0.95, "N=\(n) で recall@10 が完了条件を下回った")
        }
    }

    /// M / efConstruction / efSearch の影響調査(クラスタ構造データ、N=5,000 固定)
    @Test func パラメータ影響調査() async throws {
        let n = 5_000
        let clock = ContinuousClock()
        let (data, queries) = clusteredDataset(
            dataCount: n, queryCount: Self.queryCount,
            dimension: Self.dimension, clusters: 50, seed: 21
        )

        // recall は ID 集合の比較なので、全インデックスで同じ ID を共有する
        let ids = (0..<n).map { _ in UUID() }
        let brute = BruteForceIndex(dimension: Self.dimension)
        for (id, vector) in zip(ids, data) {
            await brute.upsert(id: id, vector: vector)
        }
        var truths: [[VectorSearchResult]] = []
        for query in queries {
            truths.append(await brute.search(query, k: 10))
        }

        // (1) M × efConstruction(efSearch=64 固定): 構築コストとグラフ質の関係
        print("[params] M, efC, build(s), recall@10, ms/query")
        for m in [8, 16, 32] {
            for efConstruction in [100, 200] {
                let hnsw = HNSWIndex(
                    dimension: Self.dimension,
                    configuration: .init(m: m, efConstruction: efConstruction, efSearch: 64, seed: 42)
                )
                let buildStart = clock.now
                for (id, vector) in zip(ids, data) {
                    await hnsw.upsert(id: id, vector: vector)
                }
                let buildTime = clock.now - buildStart

                var totalRecall = 0.0
                let searchStart = clock.now
                for (query, truth) in zip(queries, truths) {
                    totalRecall += recall(approx: await hnsw.search(query, k: 10), truth: truth)
                }
                let searchTime = (clock.now - searchStart) / Self.queryCount
                print("[params] M=\(m), efC=\(efConstruction): build=\(buildTime), "
                    + "recall=\(totalRecall / Double(Self.queryCount)), query=\(searchTime)")
            }
        }

        // (2) efSearch の掃引(M=16, efC=200 固定): recall-速度カーブ
        print("[efS] efSearch, recall@10, ms/query")
        for efSearch in [10, 16, 32, 64, 128] {
            let hnsw = HNSWIndex(
                dimension: Self.dimension,
                configuration: .init(m: 16, efConstruction: 200, efSearch: efSearch, seed: 42)
            )
            for (id, vector) in zip(ids, data) {
                await hnsw.upsert(id: id, vector: vector)
            }
            var totalRecall = 0.0
            let searchStart = clock.now
            for (query, truth) in zip(queries, truths) {
                totalRecall += recall(approx: await hnsw.search(query, k: 10), truth: truth)
            }
            let searchTime = (clock.now - searchStart) / Self.queryCount
            print("[efS] efS=\(efSearch): recall=\(totalRecall / Double(Self.queryCount)), query=\(searchTime)")
        }
    }

    /// ストレスケース: 一様ランダム高次元データ(次元の呪いで全点がほぼ等距離)。
    /// efSearch を上げると recall が 1 に近づくこと = グラフ自体は健全で、
    /// 低 recall は探索幅とデータ分布の問題であることを確認する。
    @Test func 一様ランダムデータの限界確認() async throws {
        let n = 5_000
        let clock = ContinuousClock()
        let data = uniformUnitVectors(count: n, dimension: Self.dimension, seed: 31)
        let queries = uniformUnitVectors(count: Self.queryCount, dimension: Self.dimension, seed: 32)

        let ids = (0..<n).map { _ in UUID() }
        let brute = BruteForceIndex(dimension: Self.dimension)
        for (id, vector) in zip(ids, data) {
            await brute.upsert(id: id, vector: vector)
        }
        var truths: [[VectorSearchResult]] = []
        for query in queries {
            truths.append(await brute.search(query, k: 10))
        }

        print("[uniform] efSearch, recall@10, ms/query")
        for efSearch in [64, 128, 256, 512] {
            let hnsw = HNSWIndex(
                dimension: Self.dimension,
                configuration: .init(m: 16, efConstruction: 200, efSearch: efSearch, seed: 42)
            )
            for (id, vector) in zip(ids, data) {
                await hnsw.upsert(id: id, vector: vector)
            }
            var totalRecall = 0.0
            let searchStart = clock.now
            for (query, truth) in zip(queries, truths) {
                totalRecall += recall(approx: await hnsw.search(query, k: 10), truth: truth)
            }
            let searchTime = (clock.now - searchStart) / Self.queryCount
            print("[uniform] efS=\(efSearch): recall=\(totalRecall / Double(Self.queryCount)), query=\(searchTime)")
        }
    }
}
