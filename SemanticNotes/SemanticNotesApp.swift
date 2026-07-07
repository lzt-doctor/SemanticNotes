//
//  SemanticNotesApp.swift
//  SemanticNotes
//

import SwiftData
import SwiftUI

@main
struct SemanticNotesApp: App {
    // なぜ: 「データが端末から一切出ない」ことがこのアプリの核なので、
    // 永続化はローカルの SwiftData のみ。CloudKit 同期やネットワーク機能は追加しない(CLAUDE.md の制約)。
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .modelContainer(for: [Note.self, NoteChunk.self])
    }
}
