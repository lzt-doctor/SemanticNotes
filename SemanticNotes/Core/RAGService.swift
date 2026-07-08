//
//  RAGService.swift
//  SemanticNotes
//

import Foundation
import SwiftData

/// 「検索 → 根拠の組み立て → 生成」を束ねる RAG のオーケストレータ。
/// 生成が使えない環境でも検索と根拠(sources)は必ず返す —
/// アプリの核は検索と根拠提示であり、生成はその上の薄い層という設計。
@MainActor
final class RAGService {
    struct Answer {
        /// 生成された回答。生成が使えない環境では nil
        let text: String?
        /// 生成が使えない理由(ユーザーへの案内文)。使える場合は nil
        let unavailableReason: String?
        /// 回答の根拠(検索上位チャンク、出典表示用)。生成の可否に関係なく返す
        let sources: [SearchIndexService.ChunkHit]
    }

    private let searchService: SearchIndexService
    private let generator: any AnswerGenerator

    init(searchService: SearchIndexService, generator: any AnswerGenerator) {
        self.searchService = searchService
        self.generator = generator
    }

    var generatorAvailability: AnswerGeneratorAvailability {
        generator.availability
    }

    func answer(question: String) async throws -> Answer {
        let trimmed = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return Answer(text: nil, unavailableReason: nil, sources: [])
        }

        // ① 検索: 既存の意味検索パイプラインで上位チャンクを取る
        let hits = try await searchService.topChunks(for: trimmed, limit: 5)
        guard !hits.isEmpty else {
            return Answer(text: nil, unavailableReason: nil, sources: [])
        }

        // ② 根拠の組み立て: トークン予算内に収まる上位チャンクだけを使う
        let candidates = hits.map { hit in
            RAGSource(
                id: hit.chunk.chunkID,
                noteTitle: hit.note.title,
                excerpt: hit.chunk.content,
                score: hit.score
            )
        }
        let sources = RAGPromptBuilder.selectSources(candidates)
        // 画面の出典リストは、実際に根拠へ採用したものだけを見せる(答えと出典を一致させる)
        let usedIDs = Set(sources.map(\.id))
        let usedHits = hits.filter { usedIDs.contains($0.chunk.chunkID) }

        // ③ 生成: 使えない環境では理由を添えて検索結果のみ返す(フォールバック)
        guard case .available = generator.availability else {
            if case .unavailable(let reason) = generator.availability {
                return Answer(text: nil, unavailableReason: reason, sources: usedHits)
            }
            return Answer(text: nil, unavailableReason: nil, sources: usedHits)
        }
        let text = try await generator.generate(question: trimmed, sources: sources)
        return Answer(text: text, unavailableReason: nil, sources: usedHits)
    }
}
