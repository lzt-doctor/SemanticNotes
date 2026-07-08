//
//  RAGPromptBuilder.swift
//  SemanticNotes
//

import Foundation

/// 検索上位チャンクを「出典番号付きの根拠」としてプロンプトに組み立てる。
/// 純粋な文字列変換なので単体テストしやすい。
nonisolated enum RAGPromptBuilder {
    /// 根拠に使ってよい見積もりトークンの予算。
    /// なぜ 1,500: オンデバイス LLM の文脈窓は数千トークンと小さく、
    /// 指示文・質問・回答の余白を残す必要がある。チャンクは最大400トークン
    /// 設計(Phase 1)なので、予算内で3〜5件の根拠が収まる。
    static let sourceTokenBudget = 1_500

    /// 検索順位を保ったまま、予算に収まる範囲の根拠を選ぶ。
    /// なぜ上位優先で打ち切るか: 順位はコサイン類似度による確信度なので、
    /// 下位の小さいチャンクを繰り上げるより、上位を確実に残す方が根拠の質が高い。
    static func selectSources(_ sources: [RAGSource], budget: Int = sourceTokenBudget) -> [RAGSource] {
        var selected: [RAGSource] = []
        var usedTokens = 0
        for source in sources {
            let tokens = Chunker.estimatedTokenCount(of: source.excerpt)
            if usedTokens + tokens > budget {
                break
            }
            selected.append(source)
            usedTokens += tokens
        }
        // 先頭のチャンクが単独で予算超過という異常時も、根拠ゼロでは答えられないので1件は残す
        if selected.isEmpty, let first = sources.first {
            selected.append(first)
        }
        return selected
    }

    /// LLM への指示文。「渡した抜粋だけを根拠にする」制約でハルシネーションを抑える。
    static func instructions() -> String {
        """
        あなたはユーザーの個人ノートの内容だけに基づいて答えるアシスタントです。
        ルール:
        - これから渡すノートの抜粋だけを根拠に、質問と同じ言語で簡潔に答える。
        - 抜粋に書かれていないことは推測せず、「ノートには見つかりませんでした」と答える。
        - 根拠にした抜粋の番号を [1] のような形式で回答に含める。
        """
    }

    /// 根拠と質問を1つのプロンプトへ。出典は [番号] ノート「タイトル」: 本文 の形式。
    static func prompt(question: String, sources: [RAGSource]) -> String {
        var lines: [String] = ["ノートの抜粋:"]
        for (index, source) in sources.enumerated() {
            let title = source.noteTitle.isEmpty ? "無題" : source.noteTitle
            lines.append("[\(index + 1)] ノート「\(title)」: \(source.excerpt)")
        }
        lines.append("")
        lines.append("質問: \(question)")
        return lines.joined(separator: "\n")
    }
}
