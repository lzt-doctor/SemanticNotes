//
//  VectorIndex.swift
//  SemanticNotes
//

import Foundation

struct VectorSearchResult: Sendable, Equatable {
    let id: UUID
    /// 正規化済みベクトル同士の内積 = コサイン類似度(-1〜1、大きいほど類似)
    let score: Float
}

/// ベクトル近傍検索インデックスの抽象。
/// なぜプロトコルか: 総当たり(Phase 4)と HNSW(Phase 5)を同じ検索パイプラインに
/// 差し替え可能にし、recall・速度の比較を公平に行えるようにする(CLAUDE.md の確定判断)。
nonisolated protocol VectorIndex: Sendable {
    var dimension: Int { get }
    var count: Int { get async }

    /// 同じ id での呼び出しはベクトルの置き換えになる
    func upsert(id: UUID, vector: [Float]) async
    func remove(id: UUID) async
    func removeAll() async

    /// スコア降順で上位 k 件を返す。クエリは L2 正規化済みであること。
    func search(_ query: [Float], k: Int) async -> [VectorSearchResult]
}
