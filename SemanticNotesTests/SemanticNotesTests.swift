//
//  SemanticNotesTests.swift
//  SemanticNotesTests
//

import Foundation
import SwiftData
import Testing

@testable import SemanticNotes

@MainActor
struct SemanticNotesTests {
    /// なぜインメモリ: テストごとに独立したストアを作り、
    /// 実行順序への依存とシミュレータ上の実データ汚染を防ぐ。
    private func makeContext() throws -> ModelContext {
        let configuration = ModelConfiguration(isStoredInMemoryOnly: true)
        let container = try ModelContainer(
            for: Note.self, NoteChunk.self,
            configurations: configuration
        )
        return ModelContext(container)
    }

    @Test func 新規ノートは再インデックス対象として作られる() throws {
        let repository = NoteRepository(modelContext: try makeContext())

        let note = try repository.create(title: "買い物メモ", content: "牛乳と卵を買う")

        #expect(note.needsReindexing)
        #expect(note.chunks.isEmpty)
        #expect(try repository.fetchAll().count == 1)
    }

    @Test func 本文を変更したときだけ再インデックス対象になる() throws {
        let repository = NoteRepository(modelContext: try makeContext())
        let note = try repository.create(title: "旧タイトル", content: "本文A")
        note.needsReindexing = false

        // タイトルだけの変更ではフラグは立たない(埋め込みの作り直しは不要)
        try repository.update(note, title: "新タイトル", content: "本文A")
        #expect(!note.needsReindexing)

        // 本文の変更でフラグが立つ
        try repository.update(note, title: "新タイトル", content: "本文B")
        #expect(note.needsReindexing)
    }

    @Test func ノートを削除するとチャンクも連鎖して消える() throws {
        let context = try makeContext()
        let repository = NoteRepository(modelContext: context)
        let note = try repository.create(title: "t", content: "c")
        context.insert(NoteChunk(content: "チャンク1", chunkIndex: 0, note: note))
        context.insert(NoteChunk(content: "chunk 2", chunkIndex: 1, note: note))
        try context.save()

        try repository.delete(note)

        let remaining = try context.fetch(FetchDescriptor<NoteChunk>())
        #expect(remaining.isEmpty)
    }

    @Test func 埋め込みベクトルはDataとの往復で値が保たれる() {
        let chunk = NoteChunk(content: "テスト", chunkIndex: 0)
        #expect(chunk.embeddingVector == nil)

        let vector: [Float] = [0.25, -1.0, 0.5, 3.14159]
        chunk.embeddingVector = vector

        #expect(chunk.embeddingVector == vector)
        // 保存形式は 4 バイト × 次元数(Phase 6 でインデックスサイズを見積もる根拠になる)
        #expect(chunk.embedding?.count == vector.count * MemoryLayout<Float>.size)
    }
}
