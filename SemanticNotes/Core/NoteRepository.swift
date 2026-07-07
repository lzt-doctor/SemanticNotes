//
//  NoteRepository.swift
//  SemanticNotes
//

import Foundation
import SwiftData

/// ノートの永続化操作を一手に引き受けるリポジトリ層。
/// なぜ挟むか: View から SwiftData の詳細を隠し、needsReindexing の管理を
/// 一箇所に集める。テストではインメモリの ModelContext を渡して差し替えられる。
@MainActor
final class NoteRepository {
    private let modelContext: ModelContext
    private let chunker: Chunker

    init(modelContext: ModelContext, chunker: Chunker = Chunker()) {
        self.modelContext = modelContext
        self.chunker = chunker
    }

    @discardableResult
    func create(title: String, content: String) throws -> Note {
        let note = Note(title: title, content: content)
        modelContext.insert(note)
        try modelContext.save()
        try reindexIfNeeded(note)
        return note
    }

    func fetchAll() throws -> [Note] {
        let descriptor = FetchDescriptor<Note>(
            sortBy: [SortDescriptor(\.updatedAt, order: .reverse)]
        )
        return try modelContext.fetch(descriptor)
    }

    /// なぜ本文変更のときだけフラグを立てるか: タイトルは検索対象のチャンクに
    /// 含まれないので、タイトルだけの変更でチャンクと埋め込みを作り直すのは無駄だから。
    func update(_ note: Note, title: String, content: String) throws {
        if note.content != content {
            note.needsReindexing = true
        }
        note.title = title
        note.content = content
        note.updatedAt = Date()
        try modelContext.save()
        try reindexIfNeeded(note)
    }

    func delete(_ note: Note) throws {
        modelContext.delete(note)
        try modelContext.save()
    }

    /// needsReindexing が立っているノートのチャンクを本文から作り直す。
    ///
    /// なぜ保存時に同期で行うか: 分割は軽い文字列処理なので、保存のたびに実行しても
    /// 体感に影響せず、チャンクが常に本文と整合している状態を保てる。
    /// 重い埋め込み計算はここでは行わない — Phase 3 以降で embedding == nil の
    /// チャンクだけを対象に非同期で計算する設計にする。
    ///
    /// なぜ全チャンクを作り直すか: 差分更新は「どのチャンクが変わったか」の判定が
    /// 複雑になる割に、分割自体は安価。差分化は埋め込みコストが現実になる
    /// Phase 4 で計測してから検討する。
    func reindexIfNeeded(_ note: Note) throws {
        guard note.needsReindexing else { return }

        let oldChunks = note.chunks
        for chunk in oldChunks {
            modelContext.delete(chunk)
        }
        for (index, piece) in chunker.chunk(note.content).enumerated() {
            modelContext.insert(NoteChunk(content: piece, chunkIndex: index, note: note))
        }
        note.needsReindexing = false
        try modelContext.save()
    }
}
