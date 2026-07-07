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

    init(modelContext: ModelContext) {
        self.modelContext = modelContext
    }

    @discardableResult
    func create(title: String, content: String) throws -> Note {
        let note = Note(title: title, content: content)
        modelContext.insert(note)
        try modelContext.save()
        return note
    }

    func fetchAll() throws -> [Note] {
        let descriptor = FetchDescriptor<Note>(
            sortBy: [SortDescriptor(\.updatedAt, order: .reverse)]
        )
        return try modelContext.fetch(descriptor)
    }

    /// なぜ本文変更のときだけフラグを立てるか: タイトルは検索対象のチャンクに
    /// 含まれないので、タイトルだけの変更で埋め込みを作り直すのは無駄だから。
    func update(_ note: Note, title: String, content: String) throws {
        if note.content != content {
            note.needsReindexing = true
        }
        note.title = title
        note.content = content
        note.updatedAt = Date()
        try modelContext.save()
    }

    func delete(_ note: Note) throws {
        modelContext.delete(note)
        try modelContext.save()
    }
}
