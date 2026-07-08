//
//  SearchIndexServiceTests.swift
//  SemanticNotesTests
//
//  サンプルノートとクエリはすべて自作。
//

import Foundation
import SwiftData
import Testing

@testable import SemanticNotes

/// 実モデル(git 管理外)が配置されているときだけ統合テストを動かす
private nonisolated(unsafe) let realModelAvailable: Bool =
    Bundle.main.url(forResource: "MultilingualE5Small", withExtension: "mlmodelc") != nil
        && Bundle.main.url(forResource: "tokenizer", withExtension: "json") != nil

@MainActor
private func makeContext() throws -> ModelContext {
    let configuration = ModelConfiguration(isStoredInMemoryOnly: true)
    let container = try ModelContainer(for: Note.self, NoteChunk.self, configurations: configuration)
    return ModelContext(container)
}

// MARK: - モック埋め込みでの決定的なテスト

@MainActor
struct SearchIndexServiceTests {
    /// ベクトルを固定したモックで、検索パイプライン(埋め込み→索引→ノート畳み込み)を検証する
    private func makeService(context: ModelContext, fixedVectors: [String: [Float]]) -> SearchIndexService {
        SearchIndexService(
            modelContext: context,
            embedder: MockEmbeddingService(dimension: 3, fixedVectors: fixedVectors),
            index: BruteForceIndex(dimension: 3)
        )
    }

    @Test func 意味的に近いノートが上位に返る() async throws {
        let context = try makeContext()
        let repository = NoteRepository(modelContext: context)
        try repository.create(title: "果樹", content: "りんごの栽培メモ。水やりは朝にする。")
        try repository.create(title: "車", content: "オイル交換の記録。次回は来年の春。")

        let service = makeService(context: context, fixedVectors: [
            "passage: りんごの栽培メモ。水やりは朝にする。": [1, 0, 0],
            "passage: オイル交換の記録。次回は来年の春。": [0, 1, 0],
            "query: 果物の育て方": [0.99, 0.14, 0], // 果樹側に近いクエリ
        ])
        try await service.refreshIndex()

        let hits = try await service.search("果物の育て方")

        #expect(hits.count == 2)
        #expect(hits.first?.note.title == "果樹")
        #expect(hits.first?.excerpt == "りんごの栽培メモ。水やりは朝にする。")
    }

    @Test func 削除したノートは再同期後の検索から消える() async throws {
        let context = try makeContext()
        let repository = NoteRepository(modelContext: context)
        try repository.create(title: "果樹", content: "りんごの栽培メモ。水やりは朝にする。")
        let carNote = try repository.create(title: "車", content: "オイル交換の記録。次回は来年の春。")

        let service = makeService(context: context, fixedVectors: [
            "passage: りんごの栽培メモ。水やりは朝にする。": [1, 0, 0],
            "passage: オイル交換の記録。次回は来年の春。": [0, 1, 0],
            "query: 車の整備": [0, 1, 0],
        ])
        try await service.refreshIndex()
        #expect(try await service.search("車の整備").first?.note.title == "車")

        // 削除 → 全量同期で索引からも消える(残骸が検索に出ない)
        try repository.delete(carNote)
        try await service.refreshIndex()

        let hits = try await service.search("車の整備")
        #expect(hits.allSatisfy { $0.note.title != "車" })
    }

    @Test func 同じノートの複数チャンクは最良の1件に畳まれる() async throws {
        let context = try makeContext()
        let note = Note(title: "長文", content: "")
        context.insert(note)
        context.insert(NoteChunk(content: "前半の話題", chunkIndex: 0, note: note))
        context.insert(NoteChunk(content: "後半の話題", chunkIndex: 1, note: note))
        try context.save()

        let service = makeService(context: context, fixedVectors: [
            "passage: 前半の話題": [1, 0, 0],
            "passage: 後半の話題": [0.9, 0.43, 0],
            "query: 話題": [1, 0, 0],
        ])
        try await service.refreshIndex()

        let hits = try await service.search("話題")

        #expect(hits.count == 1) // 2チャンクとも上位だがノートとしては1件
        #expect(hits.first?.excerpt == "前半の話題") // スコアが高い方が採用される
    }

    @Test func 空クエリは検索せず空を返す() async throws {
        let context = try makeContext()
        let service = makeService(context: context, fixedVectors: [:])

        #expect(try await service.search("   ").isEmpty)
    }
}

// MARK: - HNSW への差し替え(VectorIndex プロトコルの意義の証明)

@MainActor
struct HNSWDropInTests {
    /// インデックスを HNSW に差し替えても検索パイプラインがそのまま動くこと
    @Test func HNSWはBruteForceと差し替えて動く() async throws {
        let context = try makeContext()
        let repository = NoteRepository(modelContext: context)
        try repository.create(title: "果樹", content: "りんごの栽培メモ。水やりは朝にする。")
        try repository.create(title: "車", content: "オイル交換の記録。次回は来年の春。")

        let service = SearchIndexService(
            modelContext: context,
            embedder: MockEmbeddingService(dimension: 3, fixedVectors: [
                "passage: りんごの栽培メモ。水やりは朝にする。": [1, 0, 0],
                "passage: オイル交換の記録。次回は来年の春。": [0, 1, 0],
                "query: 果物の育て方": [0.99, 0.14, 0],
            ]),
            index: HNSWIndex(dimension: 3, configuration: .init(m: 8, efConstruction: 32, efSearch: 16, seed: 1))
        )
        try await service.refreshIndex()

        let hits = try await service.search("果物の育て方")
        #expect(hits.count == 2)
        #expect(hits.first?.note.title == "果樹")
    }
}

// MARK: - 実モデルでの意味検索(完了条件「実データで意味検索が動く」の証拠)

@Suite(.enabled(if: realModelAvailable, "モデル未配置(scripts/install_model.sh の実行後に有効)"))
@MainActor
struct SemanticSearchIntegrationTests {
    @Test func 実モデルで意味的クエリが正しいノートを1位に返す() async throws {
        let context = try makeContext()
        let repository = NoteRepository(modelContext: context)
        try repository.create(
            title: "カレー",
            content: "カレーの作り方。玉ねぎをよく炒めてから肉と水を加え、弱火でじっくり煮込む。"
        )
        try repository.create(
            title: "車検",
            content: "車検の記録。タイヤの溝の深さと、ブレーキパッドの残量を点検してもらった。"
        )
        try repository.create(
            title: "並行処理",
            content: "Swift の並行処理について。actor を使うと共有状態のデータ競合をコンパイル時に防げる。"
        )

        let embedder = try CoreMLEmbeddingService()
        let service = SearchIndexService(
            modelContext: context,
            embedder: embedder,
            index: BruteForceIndex(dimension: embedder.dimension)
        )
        try await service.refreshIndex()

        // 本文と同じ単語を使わない「意味的な」クエリで狙ったノートが1位になること
        let programming = try await service.search("非同期プログラミングのメモ")
        #expect(programming.first?.note.title == "並行処理", "hits: \(programming.map(\.note.title))")

        let cooking = try await service.search("料理の手順")
        #expect(cooking.first?.note.title == "カレー", "hits: \(cooking.map(\.note.title))")
    }
}
