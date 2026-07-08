//
//  SearchView.swift
//  SemanticNotes
//

import SwiftData
import SwiftUI

/// 検索画面の状態と処理(MVVM の VM)。View からロジックを分離してテスト可能にする。
@MainActor
@Observable
final class SearchViewModel {
    var query = ""
    private(set) var results: [SearchIndexService.NoteHit] = []
    private(set) var isPreparing = false
    private(set) var isSearching = false
    private(set) var statusMessage: String?
    private(set) var hasSearched = false

    private var service: SearchIndexService?

    /// 画面表示時に呼ぶ。埋め込みモデルを読み込み、インデックスを同期する。
    /// なぜ画面表示ごとか: 直前の編集を確実に反映するため。埋め込み済みチャンクは
    /// 再計算されないので、2回目以降は同期コストだけで済む。
    func prepare(modelContext: ModelContext) async {
        isPreparing = true
        defer { isPreparing = false }
        do {
            if service == nil {
                let embedder = try CoreMLEmbeddingService()
                service = SearchIndexService(
                    modelContext: modelContext,
                    embedder: embedder,
                    index: BruteForceIndex(dimension: embedder.dimension)
                )
            }
            try await service?.refreshIndex()
            statusMessage = nil
        } catch {
            // モデル未配置(scripts/install_model.sh 未実行)が典型。検索だけ無効化する
            service = nil
            statusMessage = "検索モデルを読み込めませんでした。scripts/README.md の手順でモデルを配置してください。"
        }
    }

    func runSearch() async {
        guard let service else { return }
        isSearching = true
        defer { isSearching = false }
        do {
            results = try await service.search(query)
            hasSearched = true
        } catch {
            statusMessage = "検索に失敗しました: \(error.localizedDescription)"
        }
    }
}

/// 意味検索画面。クエリを入力して確定すると、意味的に近いノートが上位から並ぶ。
struct SearchView: View {
    @Environment(\.modelContext) private var modelContext
    @State private var viewModel = SearchViewModel()

    var body: some View {
        List {
            if let message = viewModel.statusMessage {
                ContentUnavailableView(
                    "検索を利用できません",
                    systemImage: "exclamationmark.triangle",
                    description: Text(message)
                )
            } else if viewModel.hasSearched && viewModel.results.isEmpty {
                ContentUnavailableView.search(text: viewModel.query)
            } else {
                ForEach(viewModel.results) { hit in
                    NavigationLink(value: hit.note) {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(hit.note.title.isEmpty ? String(localized: "無題") : hit.note.title)
                                    .font(.headline)
                                Spacer()
                                // 類似度は開発中の手がかりとして表示(仕上げ時に見直す)
                                Text(String(format: "%.3f", hit.score))
                                    .font(.caption.monospacedDigit())
                                    .foregroundStyle(.secondary)
                            }
                            Text(hit.excerpt)
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                                .lineLimit(3)
                        }
                    }
                }
            }
        }
        .navigationTitle("意味検索")
        .searchable(
            text: $viewModel.query,
            placement: .navigationBarDrawer(displayMode: .always),
            prompt: "意味で探す(例: 先週の決定事項)"
        )
        .onSubmit(of: .search) {
            Task { await viewModel.runSearch() }
        }
        .overlay {
            if viewModel.isPreparing {
                ProgressView("インデックスを準備中…")
            } else if viewModel.isSearching {
                ProgressView()
            }
        }
        .task {
            await viewModel.prepare(modelContext: modelContext)
        }
    }
}

#Preview {
    NavigationStack {
        SearchView()
    }
    .modelContainer(for: [Note.self, NoteChunk.self], inMemory: true)
}
