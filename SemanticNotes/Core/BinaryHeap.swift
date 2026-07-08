//
//  BinaryHeap.swift
//  SemanticNotes
//

import Foundation

/// 配列ベースの二分ヒープ。HNSW の探索で「最良候補の取り出し」(類似度降順)と
/// 「最悪結果の追い出し」(類似度昇順)の両方に、比較関数を変えて使う。
/// なぜ自作か: 外部依存を増やさず、この規模(約50行)なら単体テストで完全に検証できる。
nonisolated struct BinaryHeap<Element> {
    private var elements: [Element] = []
    /// a が b より先に取り出されるべきなら true
    private let comesFirst: (Element, Element) -> Bool

    init(comesFirst: @escaping (Element, Element) -> Bool) {
        self.comesFirst = comesFirst
    }

    var isEmpty: Bool { elements.isEmpty }
    var count: Int { elements.count }
    /// 次に取り出される要素(取り出さずに見る)
    var peek: Element? { elements.first }

    mutating func push(_ element: Element) {
        elements.append(element)
        siftUp(from: elements.count - 1)
    }

    @discardableResult
    mutating func pop() -> Element? {
        guard let first = elements.first else { return nil }
        let last = elements.removeLast()
        if !elements.isEmpty {
            elements[0] = last
            siftDown(from: 0)
        }
        return first
    }

    private mutating func siftUp(from index: Int) {
        var child = index
        while child > 0 {
            let parent = (child - 1) / 2
            guard comesFirst(elements[child], elements[parent]) else { break }
            elements.swapAt(child, parent)
            child = parent
        }
    }

    private mutating func siftDown(from index: Int) {
        var parent = index
        while true {
            let left = 2 * parent + 1
            let right = left + 1
            var best = parent
            if left < elements.count, comesFirst(elements[left], elements[best]) {
                best = left
            }
            if right < elements.count, comesFirst(elements[right], elements[best]) {
                best = right
            }
            guard best != parent else { break }
            elements.swapAt(parent, best)
            parent = best
        }
    }
}
