//
//  ChunkerTests.swift
//  SemanticNotesTests
//
//  サンプルテキストはすべて自作(CLAUDE.md の制約: 既存の記事等を流用しない)。
//

import Foundation
import Testing

@testable import SemanticNotes

struct ChunkerTests {
    // MARK: - 基本形

    @Test func 空文字と空白のみの入力は空配列になる() {
        let chunker = Chunker()
        #expect(chunker.chunk("").isEmpty)
        #expect(chunker.chunk("   \n\n  \t ").isEmpty)
    }

    @Test func 短いノートは1チャンクにそのまま入る() {
        let chunker = Chunker()
        let text = "今日の気づきを一行だけ書く。"
        #expect(chunker.chunk(text) == [text])
    }

    @Test func 複数の短い段落は1チャンクへ結合される() {
        let chunker = Chunker()
        // 空行は無視され、行(段落)は改行1つで再結合される
        let text = "一行目のメモ。\n二行目のメモ。\n\n三行目のメモ。"
        #expect(chunker.chunk(text) == ["一行目のメモ。\n二行目のメモ。\n三行目のメモ。"])
    }

    // MARK: - 分割と予算

    @Test func 予算を超えると複数チャンクへ分かれる() {
        // オーバーラップを 0 にして、詰め込みロジックだけを決定的に検証する
        let chunker = Chunker(configuration: .init(targetTokens: 30, maxTokens: 50, overlapTokens: 0))
        let numerals = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
        // 「覚書X。」は見積もり4トークン → 7個で28、8個目で32となり予算30を超える
        let paragraphs = numerals.map { "覚書\($0)。" }
        let chunks = chunker.chunk(paragraphs.joined(separator: "\n"))

        #expect(chunks == [
            paragraphs[0..<7].joined(separator: "\n"),
            paragraphs[7..<10].joined(separator: "\n"),
        ])
    }

    @Test func チャンク境界では前チャンク末尾の単位が次チャンク先頭に重複する() {
        let chunker = Chunker(configuration: .init(targetTokens: 30, maxTokens: 50, overlapTokens: 20))
        let numerals = ["一", "二", "三", "四", "五"]
        // 各文は見積もり19トークン(オーバーラップ上限20に収まるサイズ)
        let sentences = numerals.map { "覚書その\($0)。今日も端末内で検索を試す。" }
        let chunks = chunker.chunk(sentences.joined(separator: "\n"))

        #expect(chunks.count >= 2)
        for index in 1..<chunks.count {
            // 次チャンクの先頭単位(1行目)は、前チャンクに含まれていた単位のはず
            let firstLine = chunks[index].split(separator: "\n").first.map(String.init) ?? ""
            #expect(!firstLine.isEmpty)
            #expect(chunks[index - 1].contains(firstLine))
        }
    }

    @Test func 一行に詰まった長い段落は文単位で分割される() {
        let chunker = Chunker(configuration: .init(targetTokens: 30, maxTokens: 50, overlapTokens: 0))
        let numerals = ["一", "二", "三", "四"]
        // 句点1つの「本当に1文」を単位にする(句点2つだと文分割でさらに割れて前提が崩れる)
        let sentences = numerals.map { "議事メモ第\($0)として決定事項を必ず記録する。" }
        // 改行なしの1段落(見積もり約84トークン > 上限50)→ 文分割が発動する
        let chunks = chunker.chunk(sentences.joined())

        #expect(chunks.count >= 2)
        let joined = chunks.joined(separator: "\n")
        for sentence in sentences {
            #expect(joined.contains(sentence))
        }
        for chunk in chunks {
            #expect(Chunker.estimatedTokenCount(of: chunk) <= 50)
        }
    }

    @Test func 句読点のない長文も上限以下へ強制分割される() {
        let chunker = Chunker(configuration: .init(targetTokens: 30, maxTokens: 50, overlapTokens: 10))
        // 句点も空白もない200文字 → 文分割が効かず、強制分割の最終手段に落ちる
        let text = String(repeating: "め", count: 200)
        let chunks = chunker.chunk(text)

        #expect(chunks.count == 4)
        for chunk in chunks {
            #expect(Chunker.estimatedTokenCount(of: chunk) <= 50)
        }
        // 強制分割では取りこぼしも重複もない(オーバーラップは単位が大きすぎて付かない)
        #expect(chunks.joined() == text)
    }

    // MARK: - トークン見積もり

    @Test func 日本語は英語より同じ文字数でトークンを多く見積もる() {
        // CJK は1文字1トークン、それ以外は4文字1トークン(切り上げ)という保守的見積もり
        #expect(Chunker.estimatedTokenCount(of: "検索設計を記録する。") == 10)
        #expect(Chunker.estimatedTokenCount(of: "abcdefghij") == 3)
        #expect(Chunker.estimatedTokenCount(of: "") == 0)
    }

    // MARK: - 性質

    @Test func 同じ入力からは同じ出力が得られる() {
        let chunker = Chunker()
        let text = """
        会議メモ: 埋め込みモデルの入力には接頭辞が要る。
        Remember: prefix every query before embedding.
        検証タスク: トークナイザの一致を必ず確認する。
        Idea: compare brute force and HNSW on the same data.
        """
        let first = chunker.chunk(text)
        let second = chunker.chunk(text)

        #expect(first == second)
        #expect(!first.isEmpty)
        // 日英どの段落も失われない
        for line in text.split(separator: "\n") {
            #expect(first.joined(separator: "\n").contains(String(line)))
        }
    }
}
