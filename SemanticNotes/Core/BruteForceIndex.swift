//
//  BruteForceIndex.swift
//  SemanticNotes
//

import Accelerate
import Foundation

/// 総当たりのベクトル検索。計算量は O(N·d)。
/// なぜ最初に総当たりか: 実装が単純で「必ず正しい」結果を返すため、
/// 後続の HNSW(近似)の recall と速度を測るときの基準線(ground truth)になる。
///
/// なぜ actor か: 検索とインデックス更新が並行しても内部配列が壊れないようにする。
actor BruteForceIndex: VectorIndex {
    nonisolated let dimension: Int

    /// 全ベクトルを row-major で1本の連続配列に平坦化して持つ。
    /// なぜ平坦化か: vDSP_mmul(行列×ベクトル)1回で全チャンクとの内積を計算でき、
    /// [[Float]] のポインタ間接参照よりキャッシュ効率がよい。
    private var storage: [Float] = []
    private var ids: [UUID] = []
    private var rowOf: [UUID: Int] = [:]

    init(dimension: Int) {
        self.dimension = dimension
    }

    var count: Int { ids.count }

    func upsert(id: UUID, vector: [Float]) {
        precondition(vector.count == dimension, "次元数が一致しません")
        if let row = rowOf[id] {
            storage.replaceSubrange(row * dimension..<(row + 1) * dimension, with: vector)
        } else {
            rowOf[id] = ids.count
            ids.append(id)
            storage.append(contentsOf: vector)
        }
    }

    func remove(id: UUID) {
        guard let row = rowOf.removeValue(forKey: id) else { return }
        let lastRow = ids.count - 1
        // 穴を作らないよう、末尾の行を削除位置へ移す(swap-remove)
        if row != lastRow {
            let lastID = ids[lastRow]
            ids[row] = lastID
            rowOf[lastID] = row
            storage.replaceSubrange(
                row * dimension..<(row + 1) * dimension,
                with: storage[lastRow * dimension..<(lastRow + 1) * dimension]
            )
        }
        ids.removeLast()
        storage.removeLast(dimension)
    }

    func removeAll() {
        storage = []
        ids = []
        rowOf = [:]
    }

    func search(_ query: [Float], k: Int) -> [VectorSearchResult] {
        precondition(query.count == dimension, "次元数が一致しません")
        let n = ids.count
        guard n > 0, k > 0 else { return [] }

        // 保存ベクトルもクエリも L2 正規化済みなので「内積 = コサイン類似度」。
        // (N×d) 行列 × (d×1) ベクトル = 全チャンクとの内積を1回で計算する
        var scores = [Float](repeating: 0, count: n)
        storage.withUnsafeBufferPointer { matrix in
            query.withUnsafeBufferPointer { vector in
                vDSP_mmul(
                    matrix.baseAddress!, 1,
                    vector.baseAddress!, 1,
                    &scores, 1,
                    vDSP_Length(n), 1, vDSP_Length(dimension)
                )
            }
        }

        // 上位 k 件の選択。N が数万規模までは全ソートで十分速い
        // (プロファイルで問題になったら部分選択アルゴリズムに置き換える)
        return scores.indices
            .sorted { scores[$0] > scores[$1] }
            .prefix(k)
            .map { VectorSearchResult(id: ids[$0], score: scores[$0]) }
    }
}
