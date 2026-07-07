//
//  ContentView.swift
//  SemanticNotes
//

import SwiftData
import SwiftUI

/// ノート一覧。Phase 4 でここに検索画面への導線を追加する予定。
struct ContentView: View {
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \Note.updatedAt, order: .reverse) private var notes: [Note]

    var body: some View {
        NavigationStack {
            List {
                ForEach(notes) { note in
                    NavigationLink(value: note) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(note.title.isEmpty ? String(localized: "無題") : note.title)
                                .font(.headline)
                            if !note.content.isEmpty {
                                Text(note.content)
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(2)
                            }
                        }
                    }
                }
                .onDelete(perform: deleteNotes)
            }
            .navigationTitle("SemanticNotes")
            .navigationDestination(for: Note.self) { note in
                NoteEditorView(note: note)
            }
            .toolbar {
                ToolbarItem {
                    Button("追加", systemImage: "plus", action: addNote)
                }
            }
            .overlay {
                if notes.isEmpty {
                    ContentUnavailableView(
                        "ノートがありません",
                        systemImage: "note.text",
                        description: Text("右上の + から作成できます")
                    )
                }
            }
        }
    }

    // 書き込みはすべてリポジトリ経由に統一する(needsReindexing の管理を一箇所に保つため)。
    // Phase 0 ではエラー表示 UI は範囲外なので try? で握りつぶす。
    private func addNote() {
        let repository = NoteRepository(modelContext: modelContext)
        _ = try? repository.create(title: "", content: "")
    }

    private func deleteNotes(at offsets: IndexSet) {
        let repository = NoteRepository(modelContext: modelContext)
        for index in offsets {
            try? repository.delete(notes[index])
        }
    }
}

/// ノート編集画面。Phase 1 以降、保存時のチャンク再生成をここに接続する。
struct NoteEditorView: View {
    @Environment(\.modelContext) private var modelContext
    private let note: Note

    // なぜローカル状態に写すか: モデルへ直接バインドすると1文字ごとに保存が走る。
    // 編集はバッファ上で行い、画面を離れるときにまとめてリポジトリ経由で保存する。
    @State private var title: String
    @State private var content: String

    init(note: Note) {
        self.note = note
        _title = State(initialValue: note.title)
        _content = State(initialValue: note.content)
    }

    var body: some View {
        Form {
            TextField("タイトル", text: $title)
                .font(.headline)
            TextEditor(text: $content)
                .frame(minHeight: 280)
        }
        .navigationTitle("ノート")
        .navigationBarTitleDisplayMode(.inline)
        .onDisappear(perform: save)
    }

    private func save() {
        // 変更がなければ保存しない(needsReindexing を不用意に立てないため)
        guard title != note.title || content != note.content else { return }
        let repository = NoteRepository(modelContext: modelContext)
        try? repository.update(note, title: title, content: content)
    }
}

#Preview {
    ContentView()
        .modelContainer(for: [Note.self, NoteChunk.self], inMemory: true)
}
