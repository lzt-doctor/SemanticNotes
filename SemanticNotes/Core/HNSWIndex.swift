//
//  HNSWIndex.swift
//  SemanticNotes
//
//  HNSW (Hierarchical Navigable Small World, Malkov & Yashunin 2016) の自作実装。
//  上の層ほどノードを確率的に間引いた多層の近傍グラフを作り、検索は最上層の
//  入口から貪欲にクエリへ近づき、第0層でビーム幅 efSearch の探索に切り替える。
//  総当たり O(N·d) に対し、経験的に O(log N · d) 程度で近似 top-k を返す。
//

import Accelerate
import Foundation

/// 探索中の候補(内部インデックスと類似度)
private nonisolated struct HNSWCandidate {
    let index: Int
    let similarity: Float
}

/// 乱数(層の割り当て用)。なぜ自作の SplitMix64 か: シードを固定すると
/// グラフ構築が決定的になり、recall のテストが再現可能になるため。
private nonisolated struct SplitMix64: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}

actor HNSWIndex: VectorIndex {
    struct Configuration {
        /// 1ノードのリンク上限(第0層のみ慣例どおり 2M まで許す)
        var m: Int
        /// 挿入時に吟味する近傍候補の数(グラフの質)
        var efConstruction: Int
        /// 検索時のビーム幅(recall と速度のつまみ)
        var efSearch: Int
        /// 層割り当ての乱数シード。固定するとテストが再現可能になる
        var seed: UInt64

        static let `default` = Configuration(m: 16, efConstruction: 200, efSearch: 64, seed: .random(in: .min ... .max))
    }

    enum HNSWError: Error {
        case corruptFile(String)
        case dimensionMismatch
    }

    nonisolated let dimension: Int
    let configuration: Configuration

    // ノードは内部では Int の連番で扱う(UUID より軽く、隣接リストが小さくなる)。
    // ベクトルは row-major の連続配列(キャッシュ効率と vDSP のため)。
    private var vectors: [Float] = []
    private var uuids: [UUID] = []
    private var levels: [Int] = []
    /// neighbors[node][layer] = 隣接ノードの内部インデックス列
    private var neighbors: [[[Int32]]] = []
    /// 生存ノードのみを引ける辞書(墓標は含まない)
    private var indexOf: [UUID: Int] = [:]
    private var deleted: [Bool] = []
    private var deletedCount = 0

    private var entryPoint: Int?
    private var topLevel = 0
    private var rng: SplitMix64
    /// 層の期待サイズ比が 1/M になる正規化係数(論文の mL = 1/ln M)
    private let levelMultiplier: Double

    init(dimension: Int, configuration: Configuration = .default) {
        self.dimension = dimension
        self.configuration = configuration
        self.rng = SplitMix64(state: configuration.seed)
        self.levelMultiplier = 1.0 / log(Double(configuration.m))
    }

    // MARK: - VectorIndex

    var count: Int { indexOf.count }

    func upsert(id: UUID, vector: [Float]) {
        precondition(vector.count == dimension, "次元数が一致しません")
        // 置き換えは「旧ノードを墓標にして新規挿入」。リンクの繋ぎ直しで
        // グラフ品質を壊すより、安全側に倒す(溜まったら再構築で回収する)
        if let existing = indexOf.removeValue(forKey: id) {
            tombstone(existing)
        }
        insert(id: id, vector: vector)
        compactIfNeeded()
    }

    func remove(id: UUID) {
        guard let existing = indexOf.removeValue(forKey: id) else { return }
        tombstone(existing)
        compactIfNeeded()
    }

    func removeAll() {
        vectors = []
        uuids = []
        levels = []
        neighbors = []
        indexOf = [:]
        deleted = []
        deletedCount = 0
        entryPoint = nil
        topLevel = 0
    }

    func search(_ query: [Float], k: Int) -> [VectorSearchResult] {
        precondition(query.count == dimension, "次元数が一致しません")
        guard let entryPoint, k > 0, !indexOf.isEmpty else { return [] }

        // 上層は貪欲降下(ビーム幅1で十分。層がまばらなので迷わない)
        var ep = entryPoint
        var layer = topLevel
        while layer > 0 {
            ep = greedyClosest(to: query, from: ep, layer: layer)
            layer -= 1
        }
        // 第0層だけ幅 efSearch のビーム探索(取りこぼしを防ぐ本体)
        let found = searchLayer(query, entry: ep, ef: max(configuration.efSearch, k), layer: 0)
        return found.prefix(k).map { VectorSearchResult(id: uuids[$0.index], score: $0.similarity) }
    }

    // MARK: - 挿入

    private func insert(id: UUID, vector: [Float]) {
        let level = randomLevel()
        let newIndex = uuids.count
        uuids.append(id)
        vectors.append(contentsOf: vector)
        levels.append(level)
        neighbors.append(Array(repeating: [], count: level + 1))
        deleted.append(false)
        indexOf[id] = newIndex

        guard let currentEntry = entryPoint else {
            entryPoint = newIndex
            topLevel = level
            return
        }

        var ep = currentEntry
        // 新ノードが住まない上層は、入口をクエリ近くまで貪欲に降下させるだけ
        if level < topLevel {
            var layer = topLevel
            while layer > level {
                ep = greedyClosest(to: vector, from: ep, layer: layer)
                layer -= 1
            }
        }
        // 新ノードが住む各層で、近傍候補を efConstruction 個吟味して接続する
        var layer = min(level, topLevel)
        while layer >= 0 {
            let candidates = searchLayer(vector, entry: ep, ef: configuration.efConstruction, layer: layer)
            let maxLinks = maxLinks(at: layer)
            let selected = selectNeighbors(candidates, max: maxLinks, query: vector)

            neighbors[newIndex][layer] = selected.map { Int32($0.index) }
            for candidate in selected {
                neighbors[candidate.index][layer].append(Int32(newIndex))
                // 相手のリンクが上限を超えたら、同じ多様性基準で刈り込む
                if neighbors[candidate.index][layer].count > maxLinks {
                    pruneLinks(of: candidate.index, layer: layer, max: maxLinks)
                }
            }
            ep = candidates.first?.index ?? ep
            layer -= 1
        }

        if level > topLevel {
            topLevel = level
            entryPoint = newIndex
        }
    }

    /// 各ノードの最高層を確率的に決める。1つ上の層に住む確率が 1/M になるため、
    /// 層のサイズが指数的に減り、上層が自然と「長距離便」になる。
    private func randomLevel() -> Int {
        let uniform = Double(rng.next() >> 11) * 0x1.0p-53 // [0, 1)
        let level = Int(-log(max(uniform, .leastNonzeroMagnitude)) * levelMultiplier)
        return min(level, 32) // 理論上の暴走ガード
    }

    private func maxLinks(at layer: Int) -> Int {
        layer == 0 ? configuration.m * 2 : configuration.m
    }

    /// 論文 Algorithm 4 の多様性ヒューリスティック。類似度順に見て、
    /// 「クエリよりも既選択ノードの方に近い」候補は飛ばす。
    /// なぜ単純な上位 M 件でないか: 同じ方向に固まった近傍ばかり選ぶと
    /// 別方向への「逃げ道」が消え、貪欲探索が局所解にはまりやすくなるため。
    private func selectNeighbors(_ candidates: [HNSWCandidate], max maxCount: Int, query: [Float]) -> [HNSWCandidate] {
        var selected: [HNSWCandidate] = []
        var skipped: [HNSWCandidate] = []
        for candidate in candidates { // candidates は類似度降順
            if selected.count >= maxCount { break }
            let dominated = selected.contains { chosen in
                similarityBetween(candidate.index, chosen.index) > candidate.similarity
            }
            if dominated {
                skipped.append(candidate)
            } else {
                selected.append(candidate)
            }
        }
        // 多様性で弾いた分は、枠が余っていれば良い順に戻す(接続不足を防ぐ)
        for candidate in skipped where selected.count < maxCount {
            selected.append(candidate)
        }
        return selected
    }

    /// 既存ノードのリンクを上限まで刈り込む(そのノード自身をクエリと見なして選び直す)
    private func pruneLinks(of node: Int, layer: Int, max maxCount: Int) {
        let nodeVector = vector(at: node)
        let candidates = neighbors[node][layer]
            .map { HNSWCandidate(index: Int($0), similarity: similarityBetween(node, Int($0))) }
            .sorted { $0.similarity > $1.similarity }
        neighbors[node][layer] = selectNeighbors(candidates, max: maxCount, query: nodeVector).map { Int32($0.index) }
    }

    // MARK: - 探索

    /// ビーム幅1の貪欲降下(上層用)。改善が止まったノードを返す。
    private func greedyClosest(to query: [Float], from start: Int, layer: Int) -> Int {
        var current = start
        var currentSimilarity = similarity(current, to: query)
        var improved = true
        while improved {
            improved = false
            for neighbor in neighbors[current][safeLayer: layer] {
                let candidate = Int(neighbor)
                let sim = similarity(candidate, to: query)
                if sim > currentSimilarity {
                    current = candidate
                    currentSimilarity = sim
                    improved = true
                }
            }
        }
        return current
    }

    /// 論文 Algorithm 2。幅 ef の候補リストを保ちながら層内を探索する。
    /// 墓標ノードは「通過はできるが結果には入れない」— 削除でグラフの道を
    /// 壊さないための tombstone 方式の要点。返り値は類似度降順。
    private func searchLayer(_ query: [Float], entry: Int, ef: Int, layer: Int) -> [HNSWCandidate] {
        var visited = [Bool](repeating: false, count: uuids.count)
        visited[entry] = true

        // candidates: 次に展開すべき最良ノード(類似度が高い順に取り出す)
        var candidates = BinaryHeap<HNSWCandidate> { $0.similarity > $1.similarity }
        // results: 現時点の上位 ef 件(最悪を先頭にして追い出せるようにする)
        var results = BinaryHeap<HNSWCandidate> { $0.similarity < $1.similarity }

        let entryCandidate = HNSWCandidate(index: entry, similarity: similarity(entry, to: query))
        candidates.push(entryCandidate)
        if !deleted[entry] {
            results.push(entryCandidate)
        }

        while let current = candidates.pop() {
            // 展開候補が「保持中の最悪」より遠ければ、それ以上の改善はない
            if results.count >= ef, let worst = results.peek, current.similarity < worst.similarity {
                break
            }
            for neighbor in neighbors[current.index][safeLayer: layer] {
                let next = Int(neighbor)
                if visited[next] { continue }
                visited[next] = true
                let sim = similarity(next, to: query)
                let worst = results.peek?.similarity ?? -.infinity
                if results.count < ef || sim > worst {
                    let candidate = HNSWCandidate(index: next, similarity: sim)
                    candidates.push(candidate)
                    if !deleted[next] {
                        results.push(candidate)
                        if results.count > ef {
                            results.pop()
                        }
                    }
                }
            }
        }

        var sorted: [HNSWCandidate] = []
        sorted.reserveCapacity(results.count)
        while let item = results.pop() {
            sorted.append(item)
        }
        return sorted.reversed() // 類似度降順
    }

    // MARK: - 墓標と再構築

    private func tombstone(_ index: Int) {
        guard !deleted[index] else { return }
        deleted[index] = true
        deletedCount += 1
    }

    /// 墓標が生存ノードの半分を超えたら作り直す。
    /// なぜ再構築か: 墓標は検索を正しく保つが、グラフの遠回りとメモリは残る。
    /// 溜まった時点でまとめて払う方が、削除のたびにリンクを繋ぎ直すより安全。
    private func compactIfNeeded() {
        guard deletedCount > 64, deletedCount * 2 > indexOf.count else { return }
        rebuild()
    }

    private func rebuild() {
        let live: [(UUID, [Float])] = indexOf.map { id, index in (id, vector(at: index)) }
        removeAll()
        for (id, vector) in live {
            insert(id: id, vector: vector)
        }
    }

    // MARK: - 永続化(独自バイナリ形式)

    private static let fileMagic: UInt32 = 0x484E_5357 // "HNSW"
    private static let fileVersion: UInt32 = 1

    /// なぜ独自バイナリか: JSON/Codable だと 1万チャンク(384次元)で数十MBに膨らみ、
    /// 読み書きも遅い。ヘッダで版数と次元を検証しつつ、そのままの形で保存する。
    func save(to url: URL) throws {
        if deletedCount > 0 { rebuild() } // 墓標は保存しない(形式を単純に保つ)

        var data = Data()
        data.appendUInt32(Self.fileMagic)
        data.appendUInt32(Self.fileVersion)
        data.appendUInt32(UInt32(dimension))
        data.appendUInt32(UInt32(configuration.m))
        data.appendUInt32(UInt32(uuids.count))
        data.appendUInt32(UInt32(topLevel))
        data.appendInt32(Int32(entryPoint ?? -1))

        for index in 0..<uuids.count {
            data.appendUUID(uuids[index])
            data.appendUInt32(UInt32(levels[index]))
            for value in vector(at: index) {
                data.appendUInt32(value.bitPattern)
            }
            for layer in neighbors[index] {
                data.appendUInt32(UInt32(layer.count))
                for neighbor in layer {
                    data.appendInt32(neighbor)
                }
            }
        }
        try data.write(to: url, options: .atomic)
    }

    init(contentsOf url: URL, configuration: Configuration = .default) throws {
        var reader = BinaryReader(data: try Data(contentsOf: url))
        guard try reader.readUInt32() == Self.fileMagic else {
            throw HNSWError.corruptFile("magic が一致しません")
        }
        guard try reader.readUInt32() == Self.fileVersion else {
            throw HNSWError.corruptFile("未対応のファイル版数です")
        }
        let dimension = Int(try reader.readUInt32())
        let m = Int(try reader.readUInt32())
        let nodeCount = Int(try reader.readUInt32())
        let topLevel = Int(try reader.readUInt32())
        let entryPoint = Int(try reader.readInt32())

        self.dimension = dimension
        var config = configuration
        config.m = m // グラフは保存時の M で作られているため、M だけはファイル側を正とする
        self.configuration = config
        self.rng = SplitMix64(state: config.seed)
        self.levelMultiplier = 1.0 / log(Double(m))

        vectors.reserveCapacity(nodeCount * dimension)
        for _ in 0..<nodeCount {
            let id = try reader.readUUID()
            let level = Int(try reader.readUInt32())
            guard level <= 64 else { throw HNSWError.corruptFile("level が異常です") }
            for _ in 0..<dimension {
                vectors.append(Float(bitPattern: try reader.readUInt32()))
            }
            var layers: [[Int32]] = []
            for _ in 0...level {
                let linkCount = Int(try reader.readUInt32())
                guard linkCount <= nodeCount else { throw HNSWError.corruptFile("リンク数が異常です") }
                var links: [Int32] = []
                links.reserveCapacity(linkCount)
                for _ in 0..<linkCount {
                    let neighbor = try reader.readInt32()
                    guard neighbor >= 0, Int(neighbor) < nodeCount else {
                        throw HNSWError.corruptFile("リンク先が範囲外です")
                    }
                    links.append(neighbor)
                }
                layers.append(links)
            }
            indexOf[id] = uuids.count
            uuids.append(id)
            levels.append(level)
            neighbors.append(layers)
            deleted.append(false)
        }
        guard indexOf.count == nodeCount else { throw HNSWError.corruptFile("UUID が重複しています") }
        guard entryPoint >= -1, entryPoint < nodeCount else { throw HNSWError.corruptFile("entryPoint が範囲外です") }
        self.entryPoint = entryPoint >= 0 ? entryPoint : nil
        self.topLevel = topLevel
    }

    // MARK: - 計測用の統計

    struct Stats: Sendable {
        let liveCount: Int
        let tombstoneCount: Int
        let totalLinks: Int
        /// ベクトル+UUID+隣接リストの概算バイト数
        let approximateBytes: Int
    }

    func stats() -> Stats {
        let links = neighbors.reduce(0) { $0 + $1.reduce(0) { $0 + $1.count } }
        let bytes = vectors.count * 4 + uuids.count * 16 + links * 4
        return Stats(
            liveCount: indexOf.count,
            tombstoneCount: deletedCount,
            totalLinks: links,
            approximateBytes: bytes
        )
    }

    // MARK: - ベクトル演算

    private func vector(at index: Int) -> [Float] {
        Array(vectors[index * dimension..<(index + 1) * dimension])
    }

    /// 正規化済みベクトル同士なので内積がそのままコサイン類似度
    private func similarity(_ index: Int, to query: [Float]) -> Float {
        var result: Float = 0
        vectors.withUnsafeBufferPointer { buffer in
            query.withUnsafeBufferPointer { q in
                vDSP_dotpr(buffer.baseAddress! + index * dimension, 1, q.baseAddress!, 1, &result, vDSP_Length(dimension))
            }
        }
        return result
    }

    private func similarityBetween(_ a: Int, _ b: Int) -> Float {
        var result: Float = 0
        vectors.withUnsafeBufferPointer { buffer in
            vDSP_dotpr(
                buffer.baseAddress! + a * dimension, 1,
                buffer.baseAddress! + b * dimension, 1,
                &result, vDSP_Length(dimension)
            )
        }
        return result
    }
}

// MARK: - バイナリ入出力の補助

private nonisolated extension Data {
    mutating func appendUInt32(_ value: UInt32) {
        Swift.withUnsafeBytes(of: value.littleEndian) { append(contentsOf: $0) }
    }

    mutating func appendInt32(_ value: Int32) {
        appendUInt32(UInt32(bitPattern: value))
    }

    mutating func appendUUID(_ uuid: UUID) {
        Swift.withUnsafeBytes(of: uuid.uuid) { append(contentsOf: $0) }
    }
}

private nonisolated struct BinaryReader {
    let data: Data
    private var offset: Int

    init(data: Data) {
        self.data = data
        self.offset = data.startIndex
    }

    mutating func readUInt32() throws -> UInt32 {
        guard offset + 4 <= data.endIndex else {
            throw HNSWIndex.HNSWError.corruptFile("ファイルが途中で終わっています")
        }
        var value: UInt32 = 0
        withUnsafeMutableBytes(of: &value) { destination in
            data.copyBytes(to: destination, from: offset..<offset + 4)
        }
        offset += 4
        return UInt32(littleEndian: value)
    }

    mutating func readInt32() throws -> Int32 {
        Int32(bitPattern: try readUInt32())
    }

    mutating func readUUID() throws -> UUID {
        guard offset + 16 <= data.endIndex else {
            throw HNSWIndex.HNSWError.corruptFile("ファイルが途中で終わっています")
        }
        var bytes = uuid_t(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        withUnsafeMutableBytes(of: &bytes) { destination in
            data.copyBytes(to: destination, from: offset..<offset + 16)
        }
        offset += 16
        return UUID(uuid: bytes)
    }
}

private nonisolated extension Array where Element == [Int32] {
    /// 層の配列外アクセスを防ぐ(そのノードが住まない層は空とみなす)
    subscript(safeLayer layer: Int) -> [Int32] {
        layer < count ? self[layer] : []
    }
}
