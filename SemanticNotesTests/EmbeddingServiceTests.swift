//
//  EmbeddingServiceTests.swift
//  SemanticNotesTests
//

import Foundation
import Testing

@testable import SemanticNotes

/// モデル成果物は git 管理外のため、CI などリソースの無い環境では自動スキップする。
/// ローカルでは scripts/convert_model.py → scripts/install_model.sh の実行後に有効になる。
private nonisolated(unsafe) let embeddingResourcesAvailable: Bool =
    Bundle.main.url(forResource: "MultilingualE5Small", withExtension: "mlmodelc") != nil
        && Bundle.main.url(forResource: "tokenizer", withExtension: "json") != nil

/// Phase 2 の validate_model.py が書き出した参照データ(検証済みの真値)
private struct ReferenceEmbeddings: Decodable {
    struct Case: Decodable {
        let id: String
        let text: String
        let tokenIDs: [Int]
        let embedding: [Float]

        enum CodingKeys: String, CodingKey {
            case id, text, embedding
            case tokenIDs = "token_ids"
        }
    }

    let dimension: Int
    let cases: [Case]
}

/// テストバンドルを特定するためのアンカー(Swift Testing には Bundle.module が無い)
private final class BundleToken {}

private func loadReference() throws -> ReferenceEmbeddings {
    let url = try #require(
        Bundle(for: BundleToken.self).url(forResource: "ReferenceEmbeddings", withExtension: "json")
    )
    return try JSONDecoder().decode(ReferenceEmbeddings.self, from: Data(contentsOf: url))
}

private func cosine(_ a: [Float], _ b: [Float]) -> Double {
    precondition(a.count == b.count)
    var dot = 0.0, normA = 0.0, normB = 0.0
    for i in 0..<a.count {
        dot += Double(a[i]) * Double(b[i])
        normA += Double(a[i]) * Double(a[i])
        normB += Double(b[i]) * Double(b[i])
    }
    return dot / (normA.squareRoot() * normB.squareRoot())
}

@Suite(.enabled(if: embeddingResourcesAvailable, "モデル未配置(scripts/install_model.sh の実行後に有効)"))
struct EmbeddingServiceTests {
    /// トークナイザの不一致は「わずかに違うベクトル」という発見しにくい壊れ方をするため、
    /// 類似度でなくトークン ID の完全一致で検証する。
    @Test func トークナイザはPython実装と完全一致する() async throws {
        let service = try CoreMLEmbeddingService()
        let reference = try loadReference()

        for testCase in reference.cases {
            let ids = await service.tokenize(testCase.text)
            #expect(ids == testCase.tokenIDs, "case: \(testCase.id)")
        }
    }

    @Test func 埋め込みはPyTorchの参照ベクトルと一致する() async throws {
        let service = try CoreMLEmbeddingService()
        let reference = try loadReference()
        #expect(service.dimension == reference.dimension)

        for testCase in reference.cases {
            let vector = try await service.embed(testCase.text)
            #expect(vector.count == reference.dimension)

            // FP16 化と実行環境の差を含めても 0.999 を超えること(PLAN の基準)
            let similarity = cosine(vector, testCase.embedding)
            #expect(similarity > 0.999, "case: \(testCase.id), cosine: \(similarity)")

            // 正規化の焼き込みが効いていること(検索時に内積=コサインの前提)
            let norm = vector.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
            #expect(abs(norm - 1) < 0.01, "case: \(testCase.id), norm: \(norm)")
        }
    }

    @Test func 埋め込み時間の計測() async throws {
        let service = try CoreMLEmbeddingService()

        // 実運用の代表値: 短い検索クエリ と チャンク上限に近い長文(自作文の繰り返し)
        let query = "query: 会議の決定事項はどこ?"
        let passage = "passage: " + String(
            repeating: "端末内での意味検索を実現するための設計判断を、後から見返せるように記録しておく。",
            count: 8
        )

        // 初回はモデルのウォームアップを含むため捨て、2回目以降を平均する
        _ = try await service.embed(query)

        let clock = ContinuousClock()
        for (label, text) in [("query(短文)", query), ("passage(長文)", passage)] {
            let iterations = 5
            let start = clock.now
            for _ in 0..<iterations {
                _ = try await service.embed(text)
            }
            let average = (clock.now - start) / iterations
            let tokens = await service.tokenize(text).count
            print("[計測] \(label): \(tokens)トークン, 平均 \(average) /回")

            // 明らかな異常(秒単位)だけを検出する緩い上限
            #expect(average < .seconds(5))
        }
    }
}

/// モックはリソース不要なので常に実行する
struct MockEmbeddingServiceTests {
    @Test func モックは決定的で正規化済みのベクトルを返す() async throws {
        let mock = MockEmbeddingService(dimension: 8)

        let first = try await mock.embed("同じ入力")
        let second = try await mock.embed("同じ入力")
        let other = try await mock.embed("違う入力")

        #expect(first == second)
        #expect(first != other)
        #expect(first.count == 8)
        let norm = first.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
        #expect(abs(norm - 1) < 1e-4)
    }

    @Test func モックは固定ベクトルを優先して返す() async throws {
        let fixed: [Float] = [1, 0, 0]
        let mock = MockEmbeddingService(dimension: 3, fixedVectors: ["query: 犬": fixed])

        #expect(try await mock.embed("query: 犬") == fixed)
        #expect(try await mock.embedQuery("犬") == fixed) // 接頭辞ヘルパー経由でも同じ
    }
}
