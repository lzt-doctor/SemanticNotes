# 開発ログ

フェーズごとに「やったこと / 迷った判断 / 計測結果」を記録する。

## Phase 0: 環境確認と CI(2026-07-07)

### やったこと

- プロジェクト名を CLAUDE.md の規約どおり `SemanticNotes` に統一(フォルダ、ターゲット、スキーム、Bundle ID `com.dudu.SemanticNotes`)。
- スターター6ファイルを「リポジトリ構成」どおりに配置:
  - `SemanticNotes/SemanticNotesApp.swift` — SwiftData の ModelContainer をセットアップ
  - `SemanticNotes/Models/Note.swift` — ノート本体(`needsReindexing` フラグ、チャンクへの cascade 削除)
  - `SemanticNotes/Models/NoteChunk.swift` — 検索の最小単位(埋め込みを Data で保持、`[Float]` 変換付き)
  - `SemanticNotes/Core/NoteRepository.swift` — 永続化操作を集約するリポジトリ層
  - `SemanticNotes/Views/ContentView.swift` — ノート一覧+編集画面(書き込みはリポジトリ経由)
  - `SemanticNotesTests/SemanticNotesTests.swift` — ユニットテスト4件(Swift Testing)
- ユニットテストターゲット `SemanticNotesTests` を追加し、共有スキームを作成(ユーザースキームは CI から見えないため共有が必須)。
- Swift 6 言語モード(`SWIFT_VERSION = 6.0`)へ変更。デプロイメントターゲットを iOS 26.0 に設定(テンプレート初期値は 26.5 だったが、「最低 iOS 26」の判断に合わせ 26.x 全域をカバー)。
- `.gitignore` を整備(ビルド生成物、`*.mlpackage` / `*.mlmodelc`、Python venv、xcuserdata)。
- GitHub Actions の CI(`.github/workflows/ci.yml`)を追加。macos-26 ランナーで Xcode 26 を選択し、iPhone 17 シミュレータで `xcodebuild test` を実行する。

### 迷った判断

- **Swift 6 言語モードにするか**: テンプレートは Swift 5 モード+approachable concurrency だった。CLAUDE.md の確定判断が「Swift 6」であり、コードベースが小さい今のうちに厳格な並行性チェックへ移行する方がコストが低いと判断。デフォルトアクター分離は MainActor のままにして段階的に扱う。
- **埋め込みベクトルの保存形式**: `[Float]` を直接 SwiftData に持たせる案もあったが、保存形式とサイズ(4バイト×384次元)が明確な `Data` を採用。読み書きは `embeddingVector` 計算プロパティに集約した。
- **エディタの保存タイミング**: モデルへの直接バインドだと1文字ごとに保存が走るため、ローカル `@State` にバッファし `onDisappear` でリポジトリ経由の一括保存にした。`needsReindexing` の管理をリポジトリ1箇所に保つ狙い。

### 計測結果

- ローカル `xcodebuild test`(iPhone 17 シミュレータ / Xcode 26.6): **4件成功・0失敗**。テスト実行 0.035 秒、テストフェーズ全体 21.7 秒(初回ビルド込み)。
- CI: GitHub リモート未設定のため未実行。リポジトリを GitHub に push した時点で Actions の初回実行を確認する。
