//
//  AnswerGenerator.swift
//  SemanticNotes
//

import Foundation

/// 回答の根拠として LLM に渡すチャンク(Sendable な素の値のみ)
nonisolated struct RAGSource: Sendable, Identifiable, Equatable {
    let id: UUID
    let noteTitle: String
    let excerpt: String
    let score: Float
}

nonisolated enum AnswerGeneratorAvailability: Sendable, Equatable {
    case available
    /// 理由はユーザーへの案内文としてそのまま表示できる形で持つ
    case unavailable(reason: String)
}

/// 根拠付き回答の生成器。なぜプロトコルか: Foundation Models は実行環境に
/// 依存する(非対応端末・Apple Intelligence 無効)ため、テストでは決定的な
/// モックへ差し替え、フォールバック経路を環境に関係なく検証できるようにする。
nonisolated protocol AnswerGenerator: Sendable {
    var availability: AnswerGeneratorAvailability { get }

    /// 質問と根拠から回答を生成する。根拠に無い内容は答えない指示を含めること。
    func generate(question: String, sources: [RAGSource]) async throws -> String
}

/// テスト用のモック。availability とレスポンスを注入できる。
nonisolated struct MockAnswerGenerator: AnswerGenerator {
    let availability: AnswerGeneratorAvailability
    let respond: @Sendable (String, [RAGSource]) -> String

    init(
        availability: AnswerGeneratorAvailability = .available,
        respond: @escaping @Sendable (String, [RAGSource]) -> String = { question, sources in
            "モック回答(質問: \(question), 根拠: \(sources.count)件)"
        }
    ) {
        self.availability = availability
        self.respond = respond
    }

    func generate(question: String, sources: [RAGSource]) async throws -> String {
        respond(question, sources)
    }
}
