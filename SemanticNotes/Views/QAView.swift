//
//  QAView.swift
//  SemanticNotes
//

import SwiftData
import SwiftUI

/// Q&A 画面の状態と処理(MVVM の VM)。
@MainActor
@Observable
final class QAViewModel {
    var question = ""
    private(set) var answer: RAGService.Answer?
    private(set) var isPreparing = false
    private(set) var isAnswering = false
    private(set) var statusMessage: String?
    /// 生成が使えない環境の案内(検索と根拠表示は使える、という文脈で出す)
    private(set) var generationNotice: String?

    private var service: RAGService?

    func prepare(modelContext: ModelContext) async {
        isPreparing = true
        defer { isPreparing = false }
        do {
            if service == nil {
                let embedder = try CoreMLEmbeddingService()
                let searchService = SearchIndexService(
                    modelContext: modelContext,
                    embedder: embedder,
                    index: BruteForceIndex(dimension: embedder.dimension)
                )
                let generator = FoundationModelAnswerGenerator()
                service = RAGService(searchService: searchService, generator: generator)
                if case .unavailable(let reason) = generator.availability {
                    generationNotice = reason
                }
                try await searchService.refreshIndex()
            }
            statusMessage = nil
        } catch {
            service = nil
            statusMessage = "検索モデルを読み込めませんでした。scripts/README.md の手順でモデルを配置してください。"
        }
    }

    func ask() async {
        guard let service, !question.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        isAnswering = true
        defer { isAnswering = false }
        do {
            answer = try await service.answer(question: question)
        } catch {
            statusMessage = "回答の生成に失敗しました: \(error.localizedDescription)"
        }
    }
}

/// ノートへの Q&A 画面。回答の下に、根拠にしたノートの抜粋を必ず表示する。
struct QAView: View {
    @Environment(\.modelContext) private var modelContext
    @State private var viewModel = QAViewModel()
    @FocusState private var questionFocused: Bool

    var body: some View {
        List {
            Section {
                HStack {
                    TextField("ノートに質問する(例: 車検で点検した項目は?)", text: $viewModel.question, axis: .vertical)
                        .focused($questionFocused)
                        .onSubmit { submit() }
                    Button("質問", systemImage: "paperplane.fill") { submit() }
                        .labelStyle(.iconOnly)
                        .disabled(viewModel.isAnswering || viewModel.question.isEmpty)
                }
            }

            if let notice = viewModel.generationNotice {
                Section {
                    Label(notice, systemImage: "info.circle")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }

            if let message = viewModel.statusMessage {
                ContentUnavailableView(
                    "Q&A を利用できません",
                    systemImage: "exclamationmark.triangle",
                    description: Text(message)
                )
            }

            if let answer = viewModel.answer {
                if let text = answer.text {
                    Section("回答") {
                        Text(text).textSelection(.enabled)
                    }
                } else if answer.sources.isEmpty {
                    Section {
                        Text("関連するノートが見つかりませんでした。")
                            .foregroundStyle(.secondary)
                    }
                }

                if !answer.sources.isEmpty {
                    // 生成が使えない環境でも、根拠(検索結果)は必ず表示する
                    Section(answer.text == nil ? "関連するノート(検索結果)" : "根拠にしたノート") {
                        ForEach(Array(answer.sources.enumerated()), id: \.element.id) { index, source in
                            NavigationLink(value: source.note) {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("[\(index + 1)] \(source.note.title.isEmpty ? String(localized: "無題") : source.note.title)")
                                        .font(.subheadline.weight(.medium))
                                    Text(source.chunk.content)
                                        .font(.footnote)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(3)
                                }
                            }
                        }
                    }
                }
            }
        }
        .navigationTitle("ノートに質問")
        .navigationBarTitleDisplayMode(.inline)
        .overlay {
            if viewModel.isPreparing {
                ProgressView("準備中…")
            } else if viewModel.isAnswering {
                ProgressView("考えています…")
            }
        }
        .task {
            await viewModel.prepare(modelContext: modelContext)
        }
    }

    private func submit() {
        questionFocused = false
        Task { await viewModel.ask() }
    }
}

#Preview {
    NavigationStack {
        QAView()
    }
    .modelContainer(for: [Note.self, NoteChunk.self], inMemory: true)
}
