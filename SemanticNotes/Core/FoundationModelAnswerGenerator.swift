//
//  FoundationModelAnswerGenerator.swift
//  SemanticNotes
//

import Foundation
import FoundationModels

/// Apple の Foundation Models(OS 同梱のオンデバイス LLM)による回答生成。
/// ネットワーク通信は発生しない(本プロジェクトの制約と整合)。
/// 利用可否は端末と設定に依存するため、必ず availability を確認してから使うこと。
nonisolated struct FoundationModelAnswerGenerator: AnswerGenerator {
    var availability: AnswerGeneratorAvailability {
        switch SystemLanguageModel.default.availability {
        case .available:
            return .available
        case .unavailable(let reason):
            return .unavailable(reason: Self.describe(reason))
        @unknown default:
            return .unavailable(reason: "回答生成を利用できません。")
        }
    }

    func generate(question: String, sources: [RAGSource]) async throws -> String {
        // セッションは質問ごとに使い捨てる。会話履歴を持たないので
        // 文脈窓を根拠と質問だけに使え、コンテキスト長の管理が単純になる
        let session = LanguageModelSession(instructions: RAGPromptBuilder.instructions())
        let response = try await session.respond(
            to: RAGPromptBuilder.prompt(question: question, sources: sources)
        )
        return response.content
    }

    /// 利用不可の理由を、そのまま画面に出せる案内文へ変換する
    private static func describe(_ reason: SystemLanguageModel.Availability.UnavailableReason) -> String {
        switch reason {
        case .deviceNotEligible:
            return "この端末は Apple Intelligence に対応していないため、回答生成は使えません。検索と根拠の表示は利用できます。"
        case .appleIntelligenceNotEnabled:
            return "設定アプリで Apple Intelligence を有効にすると、回答生成が使えるようになります。"
        case .modelNotReady:
            return "モデルを準備中です。しばらくしてからもう一度お試しください。"
        @unknown default:
            return "回答生成を利用できません。検索と根拠の表示は利用できます。"
        }
    }
}
