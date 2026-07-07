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

    @Test func Noteの初期状態は再インデックス待ちである() {
        let note = Note(title: "買い物メモ", content: "牛乳と卵を買う")
        #expect(note.needsReindexing)
        #expect(note.chunks.isEmpty)
    }

    @Test func ノート作成時にチャンクが生成されフラグが下りる() throws {
        let repository = NoteRepository(modelContext: try makeContext())

        let note = try repository.create(title: "打ち合わせ", content: "打ち合わせの記録。要点を短く残す。")

        #expect(note.chunks.count == 1)
        #expect(note.chunks.first?.content == "打ち合わせの記録。要点を短く残す。")
        #expect(!note.needsReindexing)
    }

    @Test func 本文を変更するとチャンクが作り直され孤児が残らない() throws {
        let context = try makeContext()
        let repository = NoteRepository(modelContext: context)
        let note = try repository.create(title: "メモ", content: "古い本文をここに書く。")

        try repository.update(note, title: "メモ", content: "新しい本文に書き換えた。")

        #expect(note.chunks.count == 1)
        #expect(note.chunks.first?.content == "新しい本文に書き換えた。")
        #expect(!note.needsReindexing)
        // 古いチャンクがストアに残っていないこと(孤児チェック)
        let allChunks = try context.fetch(FetchDescriptor<NoteChunk>())
        #expect(allChunks.count == note.chunks.count)
    }

    @Test func タイトルのみの変更ではチャンクを再生成しない() throws {
        let repository = NoteRepository(modelContext: try makeContext())
        let note = try repository.create(title: "旧タイトル", content: "本文は変わらない。")
        let chunkIDsBefore = Set(note.chunks.map(\.persistentModelID))

        try repository.update(note, title: "新タイトル", content: "本文は変わらない。")

        // チャンクのオブジェクトが同一のまま(作り直されていない)
        #expect(Set(note.chunks.map(\.persistentModelID)) == chunkIDsBefore)
        #expect(!note.needsReindexing)
        #expect(note.title == "新タイトル")
    }

    @Test func 長いノートは複数チャンクへ連番で分割される() throws {
        let repository = NoteRepository(modelContext: try makeContext())
        // 段落10個(1段落あたり見積もり約50トークン)→ 既定予算300を超えて複数チャンクになる
        let paragraphs = (1...10).map { index in
            "第\(index)段落は端末内検索の設計判断を記録する。埋め込みは保存時に正規化し、検索時は内積だけで比較する。"
        }
        let note = try repository.create(title: "設計ノート", content: paragraphs.joined(separator: "\n"))

        #expect(note.chunks.count >= 2)
        // chunkIndex が 0 からの連番になっている(表示や順序復元の前提)
        let indexes = note.chunks.map(\.chunkIndex).sorted()
        #expect(indexes == Array(0..<note.chunks.count))
    }

    @Test func ノートを削除するとチャンクも連鎖して消える() throws {
        let context = try makeContext()
        let repository = NoteRepository(modelContext: context)
        let note = try repository.create(title: "t", content: "削除対象の本文。チャンクも一緒に消えるはず。")
        #expect(!note.chunks.isEmpty)

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
