//
//  EvaluationBenchmarkTests.swift
//  SemanticNotesTests
//
//  Phase 6 の評価本体: 自作の日英ミニベンチマーク(200ノート・40クエリ)で
//  recall@10 / nDCG@10 / レイテンシを測る。FP16 と INT8 量子化版を比較する。
//  実行方法:
//    TEST_RUNNER_RUN_BENCHMARKS=1 xcodebuild ... test -only-testing:SemanticNotesTests/EvaluationBenchmarks
//

import Foundation
import Testing

@testable import SemanticNotes

private nonisolated(unsafe) let evaluationReady: Bool =
    ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"
        && Bundle.main.url(forResource: "MultilingualE5Small", withExtension: "mlmodelc") != nil

private struct BenchmarkDataset: Decodable {
    struct Note: Decodable {
        let id: String
        let title: String
        let content: String
    }
    struct Query: Decodable {
        let id: String
        let text: String
        let relevant: [String: Int]
    }
    let notes: [Note]
    let queries: [Query]
}

private final class BundleToken {}

private func loadDataset() throws -> BenchmarkDataset {
    let url = try #require(
        Bundle(for: BundleToken.self).url(forResource: "BenchmarkDataset", withExtension: "json")
    )
    return try JSONDecoder().decode(BenchmarkDataset.self, from: Data(contentsOf: url))
}

/// nDCG@k: 上位の取りこぼしほど重く罰する順位考慮の指標。
/// 利得は 2^等級 - 1(等級2=3.0、等級1=1.0)、位置 i の割引は log2(i+1)。
private func ndcg(ranked: [String], relevant: [String: Int], k: Int) -> Double {
    var dcg = 0.0
    for (position, id) in ranked.prefix(k).enumerated() {
        guard let grade = relevant[id] else { continue }
        dcg += (pow(2, Double(grade)) - 1) / log2(Double(position) + 2)
    }
    let idealGains = relevant.values.sorted(by: >).prefix(k)
    var idcg = 0.0
    for (position, grade) in idealGains.enumerated() {
        idcg += (pow(2, Double(grade)) - 1) / log2(Double(position) + 2)
    }
    return idcg > 0 ? dcg / idcg : 1
}

/// recall@k: ラベル付き関連ノートのうち上位 k に入った割合
private func labelRecall(ranked: [String], relevant: [String: Int], k: Int) -> Double {
    guard !relevant.isEmpty else { return 1 }
    let found = ranked.prefix(k).filter { relevant[$0] != nil }.count
    return Double(found) / Double(relevant.count)
}

@Suite(.enabled(if: evaluationReady, "TEST_RUNNER_RUN_BENCHMARKS=1 かつモデル配置済みのときだけ実行"))
struct EvaluationBenchmarks {
    @Test func 日英ミニベンチマーク評価() async throws {
        let dataset = try loadDataset()
        let clock = ContinuousClock()

        // ノート ID と UUID の対応(インデックスは UUID をキーにするため)
        var uuidOf: [String: UUID] = [:]
        var noteIDOf: [UUID: String] = [:]
        for note in dataset.notes {
            let uuid = UUID()
            uuidOf[note.id] = uuid
            noteIDOf[uuid] = note.id
        }

        let candidates: [String] = ["MultilingualE5Small", "MultilingualE5SmallInt8"]
        let variants = candidates.filter {
            Bundle.main.url(forResource: $0, withExtension: "mlmodelc") != nil
        }

        for modelName in variants {
            let service = try CoreMLEmbeddingService(modelName: modelName)
            _ = try await service.embed("query: ウォームアップ")

            // 全ノートの埋め込み(アプリと同じく本文のみを対象にする)
            let embedStart = clock.now
            var vectors: [(UUID, [Float])] = []
            for note in dataset.notes {
                let vector = try await service.embedPassage(note.content)
                vectors.append((uuidOf[note.id]!, vector))
            }
            let embedTime = clock.now - embedStart

            let brute = BruteForceIndex(dimension: service.dimension)
            let hnsw = HNSWIndex(
                dimension: service.dimension,
                configuration: .init(m: 16, efConstruction: 200, efSearch: 64, seed: 42)
            )
            for (id, vector) in vectors {
                await brute.upsert(id: id, vector: vector)
                await hnsw.upsert(id: id, vector: vector)
            }

            var bruteRecall = 0.0, bruteNDCG = 0.0
            var hnswRecall = 0.0, hnswNDCG = 0.0
            var agreement = 0.0
            var perLanguage: [String: (recall: Double, ndcg: Double, count: Int)] = [:]
            var queryEmbedTotal = Duration.zero
            var bruteSearchTotal = Duration.zero
            var hnswSearchTotal = Duration.zero
            var failures: [String] = []

            for query in dataset.queries {
                let embedQueryStart = clock.now
                let queryVector = try await service.embedQuery(query.text)
                queryEmbedTotal += clock.now - embedQueryStart

                let bruteStart = clock.now
                let bruteTop = await brute.search(queryVector, k: 10)
                bruteSearchTotal += clock.now - bruteStart

                let hnswStart = clock.now
                let hnswTop = await hnsw.search(queryVector, k: 10)
                hnswSearchTotal += clock.now - hnswStart

                let bruteRanked = bruteTop.compactMap { noteIDOf[$0.id] }
                let hnswRanked = hnswTop.compactMap { noteIDOf[$0.id] }

                let recallValue = labelRecall(ranked: bruteRanked, relevant: query.relevant, k: 10)
                let ndcgValue = ndcg(ranked: bruteRanked, relevant: query.relevant, k: 10)
                bruteRecall += recallValue
                bruteNDCG += ndcgValue
                hnswRecall += labelRecall(ranked: hnswRanked, relevant: query.relevant, k: 10)
                hnswNDCG += ndcg(ranked: hnswRanked, relevant: query.relevant, k: 10)

                // HNSW の近似劣化(真の top-10 との一致率)はラベルと独立に測る
                let truthSet = Set(bruteRanked)
                agreement += Double(hnswRanked.filter { truthSet.contains($0) }.count)
                    / Double(max(truthSet.count, 1))

                let language = query.id.hasPrefix("q-en-") ? "EN" : "JA"
                let entry = perLanguage[language] ?? (0, 0, 0)
                perLanguage[language] = (entry.recall + recallValue, entry.ndcg + ndcgValue, entry.count + 1)

                if recallValue < 1 {
                    failures.append("\(query.id): top10=\(bruteRanked.prefix(3).joined(separator: ",")) …")
                }
            }

            let n = Double(dataset.queries.count)
            print("[eval] model=\(modelName)")
            print("[eval]   passage埋め込み: 計\(embedTime)(\(embedTime / dataset.notes.count)/件)")
            print("[eval]   クエリ埋め込み: \(queryEmbedTotal / dataset.queries.count)/件")
            print("[eval]   検索: brute=\(bruteSearchTotal / dataset.queries.count), hnsw=\(hnswSearchTotal / dataset.queries.count)")
            print("[eval]   BruteForce: recall@10=\(bruteRecall / n), nDCG@10=\(bruteNDCG / n)")
            print("[eval]   HNSW:       recall@10=\(hnswRecall / n), nDCG@10=\(hnswNDCG / n), 真top10一致=\(agreement / n)")
            for (language, entry) in perLanguage.sorted(by: { $0.key < $1.key }) {
                let count = Double(entry.count)
                print("[eval]   \(language)(\(entry.count)件): recall@10=\(entry.recall / count), nDCG@10=\(entry.ndcg / count)")
            }
            if !failures.isEmpty {
                print("[eval]   取りこぼしクエリ(\(failures.count)件):")
                failures.forEach { print("[eval]     \($0)") }
            }
        }
    }
}
