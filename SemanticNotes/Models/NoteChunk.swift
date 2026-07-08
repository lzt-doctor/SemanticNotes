//
//  NoteChunk.swift
//  SemanticNotes
//

import Foundation
import SwiftData

/// セマンティック検索の最小単位。Phase 1 の Chunker が Note の本文から生成する。
/// なぜノート全体でなくチャンク単位か: 長いノートを1本のベクトルに潰すと
/// 話題が平均化されて検索精度が落ちるため、意味のまとまりごとに分けて索引する。
@Model
final class NoteChunk {
    /// ベクトルインデックス側からこのチャンクを指すための安定 ID。
    /// なぜ SwiftData の persistentModelID でなく UUID か: インデックス(Phase 5 では
    /// ディスク永続化もする)を SwiftData の内部表現から切り離しておくため。
    var chunkID: UUID

    var content: String

    /// ノート内での出現順。オーバーラップ付き分割でも元の並びを復元できるようにする。
    var chunkIndex: Int

    /// 埋め込みベクトル(multilingual-e5-small、384次元 Float32)のバイト列。
    /// なぜ Data で持つか: 保存形式(4バイト×次元数)が明確でサイズを見積もりやすく、
    /// SwiftData の配列の扱いに依存しない。読み書きは embeddingVector を使う。
    var embedding: Data?

    var note: Note?

    init(content: String, chunkIndex: Int, note: Note? = nil) {
        self.chunkID = UUID()
        self.content = content
        self.chunkIndex = chunkIndex
        self.note = note
    }

    /// [Float] と Data の相互変換。
    /// 保存するベクトルは L2 正規化済みが前提(検索時に「内積=コサイン類似度」として扱うため)。
    var embeddingVector: [Float]? {
        get {
            guard let embedding else { return nil }
            return embedding.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        set {
            embedding = newValue.map { vector in
                vector.withUnsafeBytes { Data($0) }
            }
        }
    }
}
