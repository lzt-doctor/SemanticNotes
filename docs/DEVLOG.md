# 開発ログ

フェーズごとに「やったこと / 迷った判断 / 計測結果」を記録する。

## Phase 1: チャンク分割(2026-07-07)

### やったこと

- `Core/Chunker.swift` を実装。段落(行)→ 文(NLTokenizer)→ 強制分割、という3段フォールバックで「必ず上限以下の単位」を作り、目標トークン数まで詰め込む。チャンク境界では直前チャンク末尾の単位をオーバーラップとして引き継ぐ(既定: 目標300 / 上限400 / オーバーラップ50)。
- トークン数は文字種ヒューリスティックで見積もる: CJK文字 ≈ 1文字1トークン、それ以外 ≈ 4文字1トークン(切り上げ)。実トークナイザ導入(Phase 3)までのつなぎで、多め(保守側)に見積もってモデル入力上限512を超えない方向に倒した。
- `NoteRepository.reindexIfNeeded(_:)` を追加し、create / update の保存フローで `needsReindexing` を消費してチャンクを全再生成。タイトルのみの変更では再生成しない。
- テストを16件に拡充(Chunker 単体9件+モデル/リポジトリ統合7件)。日英混在・空入力・句読点なし長文・オーバーラップ検証・孤児チャンクチェックを含む。サンプル文はすべて自作。

### 迷った判断

- **トークン見積もり**: 実トークナイザ待ち vs 文字種ヒューリスティック。後者を採用し、Phase 3 で実測トークン数と突き合わせて見積もり係数を検証する予定。
- **オーバーラップの単位**: 文字数で機械的に重ねる案もあったが、段落・文という意味の通る単位ごとに重ねる方式にした。境界をまたぐ情報の取りこぼし防止という目的に合い、テストでも検証しやすい。
- **フラグの消費タイミング**: チャンク分割は軽い文字列処理なので保存時に同期実行。重い埋め込み計算は Phase 3 以降で `embedding == nil` のチャンクだけを非同期処理する設計にする(フラグは異常終了時の取りこぼし防止も兼ねる)。
- **全再構築 vs 差分更新**: 正しさが自明な全再構築を採用。差分化は埋め込みコストが現実になる Phase 4 で計測してから判断。
- **文分割の実装**: 正規表現でなく OS 標準の NLTokenizer を採用。テスト作成中に「句点が2つある文字列は2文に割れる」という正しい挙動を確認し、テスト側の前提を修正した(実装は変更なし)。

### 計測結果

- `xcodebuild test`: **16件成功・0失敗**(テスト実行 0.040 秒)。
- 約470文字(10段落)の日本語ノート → 2チャンクに分割され、chunkIndex が連番になることを確認。

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
- CI: GitHub(lzt-doctor/SemanticNotes)へ push し、GitHub Actions の初回実行が成功(緑)であることを確認。ローカルと CI の両方でテストが緑になり、Phase 0 の完了条件を達成。
