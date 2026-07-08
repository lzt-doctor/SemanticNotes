//
//  MockEmbeddingService.swift
//  SemanticNotes
//

import Foundation

/// テスト・プレビュー用の決定的なモック。モデルもトークナイザも不要。
/// なぜ Swift 標準の Hasher を使わないか: プロセスごとにシードが変わるため
/// テストの再現性がない。FNV-1a は入力テキストだけで値が決まる。
nonisolated struct MockEmbeddingService: EmbeddingService {
    let dimension: Int

    /// 特定のテキストに返すベクトルを固定したいテスト用(類似度を制御できる)
    private let fixedVectors: [String: [Float]]

    init(dimension: Int = 384, fixedVectors: [String: [Float]] = [:]) {
        self.dimension = dimension
        self.fixedVectors = fixedVectors
    }

    func embed(_ text: String) async throws -> [Float] {
        if let fixed = fixedVectors[text] {
            return fixed
        }
        // FNV-1a で種を作り、線形合同法で次元数ぶん擬似乱数を生成 → L2 正規化。
        // 本物と同じ「正規化済みベクトル」という性質を保つ。
        var state = Self.fnv1a(text)
        var vector = (0..<dimension).map { _ -> Float in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: state >> 33)) / Float(Int32.max)
        }
        let norm = vector.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
        if norm > 0 {
            vector = vector.map { $0 / norm }
        }
        return vector
    }

    private static func fnv1a(_ text: String) -> UInt64 {
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in text.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x100000001b3
        }
        return hash
    }
}
