# SemanticNotes — プロジェクト指示書

完全オンデバイスのセマンティック検索ノートアプリ(iOS)。大学院出願(CS・AI・ソフトウェア工学系)のポートフォリオとして開発する。

## 最重要ルール(必ず守る)

- 開発者(私)がすべての設計判断を面接で説明できる状態を保つこと。docs/PLAN.md のフェーズを番号順に1つずつ進め、複数フェーズを一度に進めてはならない。
- 各フェーズ完了時には必ず停止し、次の3点を提示して私の確認を待つこと:
  (a) 変更の要約と判断理由 (b) 完了条件を満たした証拠(テスト結果・計測値) (c) 私が面接で説明できるべきポイント3つ。

## プロジェクトの性格

- 「問題設定 → 手法 → 定量評価」という小さな研究として完成させる。動くだけでは不十分で、数値で語れる評価結果が必須。
- 売りは「データが端末から一切出ないプライバシー設計」。

## 確定済みの技術判断

- Xcode 26 / Swift 6 / SwiftUI / SwiftData / 最低 iOS 26。
- MVVM + リポジトリ層。EmbeddingService と VectorIndex はプロトコルで抽象化し、モックと差し替え可能にする。
- 埋め込みモデル: intfloat/multilingual-e5-small(384次元)。E5系は入力に "query: " / "passage: " の接頭辞が必須。平均プーリングと L2 正規化は Core ML 変換時にモデル側へ焼き込む。
- トークナイザ: Hugging Face swift-transformers で tokenizer.json を読み込む。Python 実装との一致検証を必ず行う。
- 検索: 埋め込みは保存時に L2 正規化し「コサイン類似度 = 内積」として扱う。総当たり(Accelerate/vDSP)→ HNSW 自作の順で実装。
- Q&A: Foundation Models framework。利用可否チェックと、非対応環境での「検索結果表示のみ」フォールバックを必ず実装。
- テストは Swift Testing、CI は GitHub Actions。

## リポジトリ構成

- SemanticNotes/ … アプリ本体(ルートに App ファイル、Models/、Views/、Core/)
- SemanticNotesTests/ … ユニットテスト
- scripts/ … モデル変換用 Python 一式(Phase 2 で作成)
- docs/PLAN.md … フェーズ計画。作業前に必読。フェーズ完了時にチェックボックスを更新する
- docs/DEVLOG.md … 開発ログ。フェーズごとに追記
- docs/RESULTS.md … 評価結果(Phase 6 で作成)

## 作業ルール

- 新しい外部依存・大きな設計変更・ファイル削除は、事前に提案して承認を得てから実施する。
- 非自明な実装の前に、選択肢とトレードオフを2〜3行で提示する。
- すべての新機能にテストを書き、xcodebuild test が通る状態を維持する。
- コードコメントは日本語で「なぜ」を書く。
- コミットは小さく意味のある単位で行う(メッセージは日本語可)。
- フェーズ完了ごとに docs/DEVLOG.md へ「やったこと / 迷った判断 / 計測結果」を追記する。

## ビルドとテスト

- ビルド: xcodebuild -project SemanticNotes.xcodeproj -scheme SemanticNotes -destination 'platform=iOS Simulator,name=iPhone 17' build
- テスト: 上記コマンドの build を test に変える。
- シミュレータ名が手元と異なる場合は xcrun simctl list devices で確認して読み替える。

## 制約(破らない)

- ネットワーク通信を伴う機能を追加しない。CloudKit 同期も入れない。
- .mlpackage などのモデル成果物はコミットしない(.gitignore に追加し、scripts/ から再生成可能に保つ)。
- ベンチマーク用のサンプルノートやクエリはすべて自作する(既存の記事・書籍・歌詞などを流用しない)。
