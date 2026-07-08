//
//  EmbeddingService.swift
//  SemanticNotes
//

import Foundation

/// 文を埋め込みベクトルへ変換するサービスの抽象。
/// なぜプロトコルか: Core ML 実装は重くリソース必須なので、上位層(検索・UI)の
/// テストでは決定的なモックへ差し替えられるようにする(CLAUDE.md の確定判断)。
nonisolated protocol EmbeddingService: Sendable {
    /// 出力ベクトルの次元数(multilingual-e5-small は 384)
    var dimension: Int { get }

    /// テキストを L2 正規化済みベクトルへ変換する。
    /// E5 系は "query: " / "passage: " の接頭辞が必須。通常は下の
    /// embedQuery / embedPassage を使い、接頭辞の付け忘れを防ぐこと。
    func embed(_ text: String) async throws -> [Float]
}

extension EmbeddingService {
    /// 検索クエリ用の埋め込み("query: " 接頭辞を付与)
    func embedQuery(_ text: String) async throws -> [Float] {
        try await embed("query: " + text)
    }

    /// 保存チャンク用の埋め込み("passage: " 接頭辞を付与)
    func embedPassage(_ text: String) async throws -> [Float] {
        try await embed("passage: " + text)
    }
}
