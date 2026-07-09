# 配布準備(TestFlight)

初見のレビュアー(出願先の教員など)に実機で触ってもらうための TestFlight 配布チェックリスト。
このプロジェクトは通信を一切行わないため、審査上の申告はシンプルになる。

## 前提

- **Apple Developer Program**(有償・年間)への加入が必要。無償アカウントでは TestFlight 配布は不可
- App Store Connect でアプリレコードを作成(Bundle ID: `com.dudu.SemanticNotes`)

## チェックリスト

### コード署名・ビルド設定

- [ ] チーム(Apple Developer アカウント)を選択し、自動署名を有効化
- [ ] `MARKETING_VERSION`(現 1.0)と `CURRENT_PROJECT_VERSION`(ビルド番号)を配布ごとに更新
- [ ] Release 構成でアーカイブ(`Product > Archive`)

### モデルの同梱

- [ ] `scripts/convert_model.py` → `scripts/install_model.sh` を実行し、`SemanticNotes/Resources/` に
      モデルとトークナイザが入っていることを確認(git 管理外なので配布ビルド前に必須)
- [ ] 配布サイズを確認: FP16 モデル約235MB。サイズが問題ならビルドを **INT8 版**
      (`MultilingualE5SmallInt8`、約118MB)に切り替える(`CoreMLEmbeddingService(modelName:)` の既定を変更)

### アプリアイコン

- [x] `Assets.xcassets/AppIcon` に実アイコンを設定(1024px、alpha 除去済み)。原本は
      [docs/app_icon/](../docs/app_icon/)(SVG・コンセプト付き)

### プライバシー(このアプリの要)

- [ ] App Store Connect の **App Privacy** で「データを収集しない」を申告(実際に通信・収集がないため正当)
- [ ] **Export Compliance**: 標準的な暗号のみ/独自暗号なし → `ITSAppUsesNonExemptEncryption = NO` を
      Info.plist に追加すると毎回の申告をスキップできる
- [ ] Foundation Models / Apple Intelligence を使う機能があることをテスト向け説明文に明記
      (対応端末でのみ回答生成が動く旨)

### TestFlight

- [ ] アーカイブを App Store Connect にアップロード(`Distribute App > TestFlight`)
- [ ] テスト情報(連絡先・ベータ説明)を記入。内部テスターに追加
- [ ] 外部テスターに配る場合はベータ版審査(通常1営業日程度)を通す

## レビュアー向け説明文(テンプレート)

> SemanticNotes is a fully on-device semantic search app for personal notes. Add a few notes, then
> use the search screen to find them by meaning (try a query that shares no words with your note).
> The Q&A screen answers questions using an on-device LLM (requires an Apple-Intelligence-capable
> device); on other devices it falls back to showing the supporting notes. The app makes no network
> connections — you can verify this in Airplane Mode.

## 既知の残タスク

- 対応実機での Foundation Models 生成の最終確認(Phase 7 の残タスク)
- CI の `actions/checkout` が出す Node ランタイムの非推奨警告(動作影響なし。将来のアクション更新で解消)
