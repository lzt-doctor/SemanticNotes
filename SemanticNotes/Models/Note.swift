//
//  Note.swift
//  SemanticNotes
//

import Foundation
import SwiftData

/// ユーザーが作成・編集するノート本体。
/// 検索の最小単位は NoteChunk だが、UI 上の操作対象と検索結果の表示単位はこの Note。
@Model
final class Note {
    var title: String
    var content: String
    var createdAt: Date
    var updatedAt: Date

    /// なぜ: 本文が変わったノートだけを再インデックス対象として拾うためのフラグ。
    /// リポジトリが保存時に消費する(チャンク再生成が完了したら false に戻す)。
    /// 永続化しておくことで、保存と再インデックスの間にアプリが落ちても
    /// 「未処理のノート」を後から拾い直せる。
    var needsReindexing: Bool

    /// なぜ cascade: ノート削除後にチャンクが残ると、存在しないノートが
    /// 検索結果に出てしまう。ノートと運命共同体にする。
    @Relationship(deleteRule: .cascade, inverse: \NoteChunk.note)
    var chunks: [NoteChunk]

    init(title: String, content: String) {
        self.title = title
        self.content = content
        let now = Date()
        self.createdAt = now
        self.updatedAt = now
        // 新規ノートは当然まだインデックスされていない
        self.needsReindexing = true
        self.chunks = []
    }
}
