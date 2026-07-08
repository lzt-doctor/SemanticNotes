//
//  SearchIndexService.swift
//  SemanticNotes
//

import Foundation
import SwiftData

/// 「チャンクの埋め込み管理 → インデックス同期 → 意味検索」を束ねるサービス。
/// EmbeddingService / VectorIndex はプロトコルで受け取るので、テストでは
/// モック埋め込みと組み合わせて決定的に検証できる。
@MainActor
final class SearchIndexService {
    /// 検索結果(ノート単位)。excerpt は最も類似したチャンクの本文。
    struct NoteHit: Identifiable {
        let note: Note
        let excerpt: String
        let score: Float
        var id: PersistentIdentifier { note.persistentModelID }
    }

    private let modelContext: ModelContext
    private let embedder: any EmbeddingService
    private let index: any VectorIndex

    init(modelContext: ModelContext, embedder: any EmbeddingService, index: any VectorIndex) {
        self.modelContext = modelContext
        self.embedder = embedder
        self.index = index
    }

    /// ストアの全チャンクとインデックスを同期する。
    /// なぜ全量同期か: 総当たりインデックスの再構築はメモリコピーだけで安価なので、
    /// 削除・編集の追従漏れ(消したノートが検索に出る等)を構造的に無くせる。
    /// 高コストな埋め込み計算だけは既存ベクトルを再利用し、未計算のチャンクに限る。
    /// - Returns: 新たに埋め込みを計算したチャンク数
    @discardableResult
    func refreshIndex() async throws -> Int {
        await index.removeAll()
        let chunks = try modelContext.fetch(FetchDescriptor<NoteChunk>())

        var embeddedCount = 0
        for chunk in chunks {
            let vector: [Float]
            if let existing = chunk.embeddingVector {
                vector = existing
            } else {
                // 重い処理はここだけ。actor(Core ML)側で実行され UI は止まらない
                vector = try await embedder.embedPassage(chunk.content)
                chunk.embeddingVector = vector
                embeddedCount += 1
            }
            await index.upsert(id: chunk.chunkID, vector: vector)
        }
        if embeddedCount > 0 {
            try modelContext.save()
        }
        return embeddedCount
    }

    /// 意味検索。クエリに "query: " を付与して埋め込み、チャンク単位の上位候補を
    /// ノート単位へ畳んで返す(1ノートにつき最良チャンクのスコアを採用)。
    func search(_ text: String, limit: Int = 10) async throws -> [NoteHit] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        let queryVector = try await embedder.embedQuery(trimmed)
        // ノート単位に畳むと件数が減るため、チャンク候補は表示件数より多めに取る
        let hits = await index.search(queryVector, k: max(limit * 5, 30))
        guard !hits.isEmpty else { return [] }

        let hitIDs = hits.map(\.id)
        let chunks = try modelContext.fetch(
            FetchDescriptor<NoteChunk>(predicate: #Predicate { hitIDs.contains($0.chunkID) })
        )
        let chunkByID = Dictionary(uniqueKeysWithValues: chunks.map { ($0.chunkID, $0) })

        var bestPerNote: [PersistentIdentifier: NoteHit] = [:]
        for hit in hits {
            // ストアに存在しない ID(万一の索引の残骸)はここで無害化される
            guard let chunk = chunkByID[hit.id], let note = chunk.note else { continue }
            let noteID = note.persistentModelID
            if let current = bestPerNote[noteID], current.score >= hit.score { continue }
            bestPerNote[noteID] = NoteHit(note: note, excerpt: chunk.content, score: hit.score)
        }
        return bestPerNote.values
            .sorted { $0.score > $1.score }
            .prefix(limit)
            .map { $0 }
    }
}
