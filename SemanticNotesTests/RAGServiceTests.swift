//
//  RAGServiceTests.swift
//  SemanticNotesTests
//

import Foundation
import SwiftData
import Testing

@testable import SemanticNotes

/// Foundation Models が使える環境(対応実機など)でだけ実生成テストを動かす。
/// なぜ SKIP_FM_TESTS も見るか: GitHub Actions のランナーは availability を
/// 「利用可能」と報告するのに、ヘッドレス環境では実際の生成が GenerationError で
/// 失敗する。availability API だけでは実行可否を判定しきれないため、
/// CI 側(ci.yml)から明示的にスキップさせる。
private nonisolated(unsafe) let foundationModelAvailable: Bool =
    FoundationModelAnswerGenerator().availability == .available
        && ProcessInfo.processInfo.environment["SKIP_FM_TESTS"] != "1"

struct RAGPromptBuilderTests {
    private func source(_ id: String, title: String, excerpt: String) -> RAGSource {
        RAGSource(id: UUID(), noteTitle: title, excerpt: excerpt, score: 0.5)
    }

    @Test func プロンプトは出典番号とタイトル付きで組み立てられる() {
        let sources = [
            source("a", title: "車検", excerpt: "タイヤの溝とブレーキパッドを点検した。"),
            source("b", title: "", excerpt: "オイル交換は来年の春。"),
        ]
        let prompt = RAGPromptBuilder.prompt(question: "車で点検した項目は?", sources: sources)

        #expect(prompt.contains("[1] ノート「車検」: タイヤの溝とブレーキパッドを点検した。"))
        #expect(prompt.contains("[2] ノート「無題」: オイル交換は来年の春。"))
        #expect(prompt.contains("質問: 車で点検した項目は?"))
        // 指示文には「抜粋だけを根拠にする」制約が入っている(ハルシネーション対策の要)
        #expect(RAGPromptBuilder.instructions().contains("抜粋だけを根拠に"))
    }

    @Test func 根拠はトークン予算内で上位優先に選ばれる() {
        // 1件あたり見積もり600トークン(日本語600文字)→ 予算1500では2件まで
        let long = String(repeating: "設計判断を記録する。", count: 60)
        let sources = (0..<5).map { source("s\($0)", title: "メモ\($0)", excerpt: long) }

        let selected = RAGPromptBuilder.selectSources(sources)

        #expect(selected.count == 2)
        // 検索順位(スコア降順)の先頭が必ず残る
        #expect(selected.map(\.id) == Array(sources.prefix(2)).map(\.id))
    }

    @Test func 先頭の根拠が予算を超えていても1件は残す() {
        let huge = String(repeating: "あ", count: 2_000)
        let sources = [source("s0", title: "長大", excerpt: huge)]

        let selected = RAGPromptBuilder.selectSources(sources)

        #expect(selected.count == 1)
    }
}

@MainActor
struct RAGServiceTests {
    private func makeContext() throws -> ModelContext {
        let configuration = ModelConfiguration(isStoredInMemoryOnly: true)
        let container = try ModelContainer(for: Note.self, NoteChunk.self, configurations: configuration)
        return ModelContext(container)
    }

    /// 果樹と車の2ノートを持つ検索サービス(モック埋め込みで決定的)
    private func makeSearchService(context: ModelContext) throws -> SearchIndexService {
        let repository = NoteRepository(modelContext: context)
        try repository.create(title: "果樹", content: "りんごの栽培メモ。水やりは朝にする。")
        try repository.create(title: "車", content: "オイル交換の記録。次回は来年の春。")
        return SearchIndexService(
            modelContext: context,
            embedder: MockEmbeddingService(dimension: 3, fixedVectors: [
                "passage: りんごの栽培メモ。水やりは朝にする。": [1, 0, 0],
                "passage: オイル交換の記録。次回は来年の春。": [0, 1, 0],
                "query: 車の整備はいつ?": [0.1, 0.99, 0],
            ]),
            index: BruteForceIndex(dimension: 3)
        )
    }

    @Test func 生成が使えない環境では検索結果のみ返る() async throws {
        let context = try makeContext()
        let searchService = try makeSearchService(context: context)
        try await searchService.refreshIndex()
        let service = RAGService(
            searchService: searchService,
            generator: MockAnswerGenerator(availability: .unavailable(reason: "テスト用の理由"))
        )

        let answer = try await service.answer(question: "車の整備はいつ?")

        #expect(answer.text == nil)
        #expect(answer.unavailableReason == "テスト用の理由")
        // フォールバックでも根拠(検索結果)は返り、最上位は車のノート
        #expect(!answer.sources.isEmpty)
        #expect(answer.sources.first?.note.title == "車")
    }

    @Test func 生成が使える環境では根拠付きの回答が返る() async throws {
        let context = try makeContext()
        let searchService = try makeSearchService(context: context)
        try await searchService.refreshIndex()
        let service = RAGService(
            searchService: searchService,
            generator: MockAnswerGenerator { question, sources in
                "質問「\(question)」に根拠\(sources.count)件で回答 [1]"
            }
        )

        let answer = try await service.answer(question: "車の整備はいつ?")

        #expect(answer.text == "質問「車の整備はいつ?」に根拠2件で回答 [1]")
        #expect(answer.unavailableReason == nil)
        #expect(answer.sources.count == 2)
        #expect(answer.sources.count <= 5) // 根拠は上位5チャンクまで
    }

    @Test func 関連ノートが無いときは回答せず空を返す() async throws {
        let context = try makeContext()
        let searchService = SearchIndexService(
            modelContext: context,
            embedder: MockEmbeddingService(dimension: 3),
            index: BruteForceIndex(dimension: 3)
        )
        let service = RAGService(
            searchService: searchService,
            generator: MockAnswerGenerator()
        )

        let answer = try await service.answer(question: "何もないはず")
        #expect(answer.text == nil)
        #expect(answer.sources.isEmpty)

        let empty = try await service.answer(question: "   ")
        #expect(empty.text == nil)
        #expect(empty.sources.isEmpty)
    }

    /// この実行環境での Foundation Models の利用可否を記録する(常に成功)。
    /// シミュレータでは unavailable になることが多く、その場合の案内文も確認できる。
    @Test func 生成の利用可否を確認する() {
        let availability = FoundationModelAnswerGenerator().availability
        switch availability {
        case .available:
            print("[FM] このEnvironmentでは Foundation Models が利用可能")
        case .unavailable(let reason):
            print("[FM] 利用不可(フォールバック動作): \(reason)")
            #expect(!reason.isEmpty)
        }
    }
}

/// 対応環境(Apple Intelligence 有効な実機など)でのみ動く実生成テスト
@Suite(.enabled(if: foundationModelAvailable, "Foundation Models が利用可能な環境でのみ実行"))
struct FoundationModelIntegrationTests {
    @Test func 実モデルで根拠に基づく回答が生成される() async throws {
        let generator = FoundationModelAnswerGenerator()
        let sources = [
            RAGSource(
                id: UUID(),
                noteTitle: "車検",
                excerpt: "車検の記録。タイヤの溝の深さと、ブレーキパッドの残量を点検してもらった。",
                score: 0.9
            )
        ]
        let answer = try await generator.generate(question: "車検では何を点検した?", sources: sources)
        print("[FM] 生成された回答: \(answer)")
        #expect(!answer.isEmpty)
    }
}
