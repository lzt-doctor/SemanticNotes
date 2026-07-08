# scripts — モデル変換パイプライン

[intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) を
iOS アプリで使う Core ML モデル(.mlpackage)へ変換し、変換の正しさを検証する。

## 前提

- macOS(Apple Silicon)、システムの Python 3.9 以上(`python3 --version` で確認)
- 初回はモデル取得のためにネットワーク接続が必要(約 470 MB、`~/.cache/huggingface` にキャッシュされる)
- huggingface.co に繋がりにくい環境では `export HF_ENDPOINT=https://hf-mirror.com` を設定してから実行する

## 手順(再現方法)

```bash
cd scripts
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python convert_model.py    # 変換: output/MultilingualE5Small.mlpackage を生成
python validate_model.py   # 検証: PyTorch との一致確認 + 参照ベクトル JSON の書き出し
```

## 各スクリプトがやること

### convert_model.py

1. Hugging Face からモデルとトークナイザを取得
2. **平均プーリング(attention mask 考慮)+ L2 正規化をモデル側へ焼き込んだ**ラッパーを TorchScript へトレース
3. Core ML へ変換して `output/MultilingualE5Small.mlpackage` を保存
   - **FP16**: 重みとオペを半精度化。サイズ約半分・Apple シリコンで高速、埋め込みの品質低下はコサイン類似度で検証可能な範囲
   - **可変長入力(1〜512トークン)**: 短い検索クエリを固定 512 パディングで処理する無駄を避ける
4. `output/tokenizer/` に tokenizer.json 等を保存(Phase 3 でアプリに同梱する)

### validate_model.py

1. 日英 ×("query: " / "passage: ")の自作8文で、PyTorch(FP32)と Core ML(FP16)の出力の
   **コサイン類似度 > 0.999** を確認(下回ると非ゼロ終了)
2. **パディング不変性**を確認(パディングを足しても出力が変わらない = プーリングが mask を正しく使っている)
3. Phase 3 の Swift テスト用に `../SemanticNotesTests/Resources/ReferenceEmbeddings.json` を書き出す
   (入力文・トークン ID 列・FP32 参照ベクトル。トークン ID は swift-transformers の
   トークナイザ一致検証にも使う)

## 設計判断の要点(面接用メモ)

- **プーリングと正規化を焼き込む理由**: アプリ側の実装を「トークン化 → 推論 → ベクトル保存」に
  最小化し、数値誤りの入り込む場所を減らす。保存ベクトルが正規化済みなら検索時は内積だけで
  コサイン類似度になり、vDSP の 1 回の dot product で済む。
- **FP16 の影響**: サイズ約 1/2(約 470MB → 約 235MB)。誤差は検証で定量化(コサイン > 0.999)。
- **モデル成果物はコミットしない**: `.gitignore` 済み。このディレクトリから常に再生成できる状態を保つ。

## 成果物

| パス | 内容 | git 管理 |
|---|---|---|
| `output/MultilingualE5Small.mlpackage` | FP16 Core ML モデル | しない(再生成可能) |
| `output/tokenizer/` | tokenizer.json ほか | しない(再生成可能) |
| `../SemanticNotesTests/Resources/ReferenceEmbeddings.json` | 検証済み参照ベクトル | する(テストの入力) |
