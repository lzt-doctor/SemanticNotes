#!/bin/bash
# 変換済みモデルとトークナイザをアプリの Resources へ配置する。
# なぜコピーで運用するか: モデル成果物は git 管理外(CLAUDE.md の制約)なので、
# clone 直後は convert_model.py → このスクリプト、の順で再生成・配置する。
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d output/MultilingualE5Small.mlpackage ]; then
  echo "output/MultilingualE5Small.mlpackage がありません。先に convert_model.py を実行してください。" >&2
  exit 1
fi

DEST="../SemanticNotes/Resources"
mkdir -p "$DEST"
rm -rf "$DEST/MultilingualE5Small.mlpackage"
cp -R output/MultilingualE5Small.mlpackage "$DEST/"
cp output/tokenizer/tokenizer.json "$DEST/"
cp output/tokenizer/tokenizer_config.json "$DEST/"
if [ -f output/tokenizer/special_tokens_map.json ]; then
  cp output/tokenizer/special_tokens_map.json "$DEST/"
fi

echo "配置完了: $DEST"
ls -lh "$DEST"
