//
//  Chunker.swift
//  SemanticNotes
//

import Foundation
import NaturalLanguage

/// ノート本文を検索用チャンクへ分割する。純粋なテキスト変換なので単体テストしやすい。
///
/// 方針: 段落(行)を基本単位とし、上限を超える段落だけ文単位へ、句読点のない
/// 極端に長い文だけ強制分割する3段フォールバック。予算(targetTokens)まで単位を
/// 詰め、チャンク境界では直前チャンク末尾の単位をオーバーラップとして重複させる。
/// なぜ nonisolated か: 状態を持たない純粋な処理で、アクター外(RAG のプロンプト
/// 構築など)からも同期的に使うため。
nonisolated struct Chunker {
    struct Configuration {
        /// 1チャンクに詰める見積もりトークン数の目標(超えそうになったら区切る)
        var targetTokens: Int
        /// 単一の段落・文をさらに分割する上限。
        /// なぜ 400: e5-small の入力上限 512 に対し "passage: " 接頭辞と特殊トークンの
        /// 余裕を残しつつ、PLAN の目安(200〜400トークン)の上限に合わせた。
        var maxTokens: Int
        /// チャンク境界で前チャンクから引き継ぐ見積もりトークン数の上限
        var overlapTokens: Int

        static let `default` = Configuration(targetTokens: 300, maxTokens: 400, overlapTokens: 50)
    }

    let configuration: Configuration

    init(configuration: Configuration = .default) {
        self.configuration = configuration
    }

    /// 本文をチャンク文字列の配列へ分割する。空・空白のみの入力は空配列を返す。
    func chunk(_ text: String) -> [String] {
        let units = makeUnits(from: text)
        guard !units.isEmpty else { return [] }
        return pack(units)
    }

    // MARK: - トークン数の見積もり

    /// multilingual-e5-small(XLM-R 系 SentencePiece)のトークン数を文字種から見積もる。
    /// なぜヒューリスティックか: 実トークナイザの導入は Phase 3。それまでは
    /// 「CJK 文字≈1文字1トークン、それ以外≈4文字1トークン」という多め(保守側)の
    /// 見積もりで、モデルの入力上限 512 を超えない方向へ倒す。Phase 3 で実測と突き合わせる。
    static func estimatedTokenCount(of text: String) -> Int {
        var cjkCount = 0
        var otherCount = 0
        for scalar in text.unicodeScalars {
            if isCJK(scalar) {
                cjkCount += 1
            } else if !CharacterSet.whitespacesAndNewlines.contains(scalar) {
                otherCount += 1
            }
        }
        return cjkCount + Int((Double(otherCount) / 4.0).rounded(.up))
    }

    private static func isCJK(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x3000...0x303F, // CJK 記号と句読点
             0x3040...0x30FF, // ひらがな・カタカナ
             0x3400...0x4DBF, // CJK 統合漢字拡張 A
             0x4E00...0x9FFF, // CJK 統合漢字
             0xF900...0xFAFF, // CJK 互換漢字
             0xFF00...0xFFEF: // 全角英数・半角カナ
            return true
        default:
            return false
        }
    }

    // MARK: - 単位への分解

    /// 「必ず maxTokens 以下」の単位列を作る。段落 → 文 → 強制分割の順に細かくする。
    private func makeUnits(from text: String) -> [String] {
        let paragraphs = text
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }

        var units: [String] = []
        for paragraph in paragraphs {
            if Self.estimatedTokenCount(of: paragraph) <= configuration.maxTokens {
                units.append(paragraph)
                continue
            }
            for sentence in Self.sentences(in: paragraph) {
                if Self.estimatedTokenCount(of: sentence) <= configuration.maxTokens {
                    units.append(sentence)
                } else {
                    units.append(contentsOf: hardSplit(sentence))
                }
            }
        }
        return units
    }

    /// 文分割は NaturalLanguage の NLTokenizer に任せる。
    /// なぜ正規表現でないか: 「e.g.」のような略語や日英混在の句読点の扱いを
    /// 自前ルールで網羅するより、OS 標準の言語処理の方が頑健なため。
    private static func sentences(in paragraph: String) -> [String] {
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = paragraph
        var result: [String] = []
        tokenizer.enumerateTokens(in: paragraph.startIndex..<paragraph.endIndex) { range, _ in
            let sentence = String(paragraph[range]).trimmingCharacters(in: .whitespaces)
            if !sentence.isEmpty {
                result.append(sentence)
            }
            return true
        }
        return result
    }

    /// 句読点のない極端に長い文のための最終手段。見積もりトークンが上限に達するごとに切る。
    private func hardSplit(_ sentence: String) -> [String] {
        var pieces: [String] = []
        var current = ""
        var cjkCount = 0
        var otherCount = 0

        for character in sentence {
            var charCJK = 0
            var charOther = 0
            for scalar in character.unicodeScalars {
                if Self.isCJK(scalar) {
                    charCJK += 1
                } else if !CharacterSet.whitespacesAndNewlines.contains(scalar) {
                    charOther += 1
                }
            }
            let nextEstimate = (cjkCount + charCJK)
                + Int((Double(otherCount + charOther) / 4.0).rounded(.up))
            if nextEstimate > configuration.maxTokens, !current.isEmpty {
                pieces.append(current)
                current = ""
                cjkCount = 0
                otherCount = 0
            }
            current.append(character)
            cjkCount += charCJK
            otherCount += charOther
        }
        if !current.isEmpty {
            pieces.append(current)
        }
        return pieces
    }

    // MARK: - チャンクへの詰め込み

    /// 単位列を予算どおりに詰めてチャンク文字列にする。
    /// なぜ単位ごとオーバーラップするか: 文字数で機械的に重ねると文の断片が生まれる。
    /// 意味の通る単位(段落・文)で重ねる方が境界の取りこぼし防止という目的に合う。
    private func pack(_ units: [String]) -> [String] {
        var chunks: [String] = []
        var current: [String] = []
        var currentTokens = 0

        for unit in units {
            let unitTokens = Self.estimatedTokenCount(of: unit)
            if !current.isEmpty, currentTokens + unitTokens > configuration.targetTokens {
                chunks.append(current.joined(separator: "\n"))

                // 直前チャンクの末尾から overlapTokens に収まる範囲の単位を引き継ぐ
                var overlap: [String] = []
                var overlapTokens = 0
                for previous in current.reversed() {
                    let tokens = Self.estimatedTokenCount(of: previous)
                    guard overlapTokens + tokens <= configuration.overlapTokens else { break }
                    overlap.insert(previous, at: 0)
                    overlapTokens += tokens
                }
                // オーバーラップを足すと上限を超えるなら引き継がない(上限厳守を優先)
                if overlapTokens + unitTokens > configuration.maxTokens {
                    overlap = []
                    overlapTokens = 0
                }
                current = overlap
                currentTokens = overlapTokens
            }
            current.append(unit)
            currentTokens += unitTokens
        }
        if !current.isEmpty {
            chunks.append(current.joined(separator: "\n"))
        }
        return chunks
    }
}
