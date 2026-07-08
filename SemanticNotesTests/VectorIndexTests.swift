//
//  VectorIndexTests.swift
//  SemanticNotesTests
//

import Foundation
import Testing

@testable import SemanticNotes

struct VectorIndexTests {
    private let a = UUID()
    private let b = UUID()
    private let c = UUID()

    /// 手計算で答え合わせできる3次元の単位ベクトルでインデックスを組む
    private func makeIndex() async -> BruteForceIndex {
        let index = BruteForceIndex(dimension: 3)
        await index.upsert(id: a, vector: [1, 0, 0])
        await index.upsert(id: b, vector: [0.8, 0.6, 0]) // ノルム1(0.64+0.36)
        await index.upsert(id: c, vector: [0, 1, 0])
        return index
    }

    @Test func 上位k件をスコア降順で返す() async {
        let index = await makeIndex()

        let results = await index.search([1, 0, 0], k: 2)

        #expect(results.map(\.id) == [a, b])
        #expect(abs(results[0].score - 1.0) < 1e-6)
        #expect(abs(results[1].score - 0.8) < 1e-6)
    }

    @Test func 同じIDのupsertはベクトルを置き換える() async {
        let index = await makeIndex()

        await index.upsert(id: a, vector: [0, 0, 1])

        #expect(await index.count == 3)
        let results = await index.search([0, 0, 1], k: 1)
        #expect(results.first?.id == a)
    }

    @Test func 削除後も残りのベクトルが正しく検索できる() async {
        let index = await makeIndex()

        // swap-remove で末尾(c)が a の位置へ移動するケース
        await index.remove(id: a)

        #expect(await index.count == 2)
        let forC = await index.search([0, 1, 0], k: 1)
        #expect(forC.first?.id == c)
        let forB = await index.search([0.8, 0.6, 0], k: 2)
        #expect(forB.map(\.id) == [b, c])
    }

    @Test func 空インデックスと過大なkでも安全に動く() async {
        let empty = BruteForceIndex(dimension: 3)
        #expect(await empty.search([1, 0, 0], k: 10).isEmpty)

        let index = await makeIndex()
        let results = await index.search([1, 0, 0], k: 100)
        #expect(results.count == 3) // 保持数までしか返さない
    }

    /// PLAN の計測項目: N = 1,000 / 5,000 / 10,000 での検索レイテンシ。
    /// 埋め込みモデルは不要(ランダムな単位ベクトル)なので CI でも動く。
    @Test func 総当たり検索のレイテンシ計測() async {
        let dimension = 384

        for n in [1_000, 5_000, 10_000] {
            let index = BruteForceIndex(dimension: dimension)
            for _ in 0..<n {
                await index.upsert(id: UUID(), vector: Self.randomUnitVector(dimension))
            }
            let query = Self.randomUnitVector(dimension)
            _ = await index.search(query, k: 10) // ウォームアップ

            let iterations = 20
            let clock = ContinuousClock()
            let start = clock.now
            for _ in 0..<iterations {
                _ = await index.search(query, k: 10)
            }
            let average = (clock.now - start) / iterations
            print("[計測] BruteForceIndex N=\(n), d=\(dimension): 平均 \(average) /回")

            // 明らかな異常だけ検出する緩い上限(実測値は DEVLOG に記録)
            #expect(average < .seconds(1))
        }
    }

    private static func randomUnitVector(_ dimension: Int) -> [Float] {
        var vector = (0..<dimension).map { _ in Float.random(in: -1...1) }
        let norm = vector.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
        if norm > 0 {
            vector = vector.map { $0 / norm }
        }
        return vector
    }
}
