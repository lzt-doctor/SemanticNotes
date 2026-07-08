#!/usr/bin/env python3
"""multilingual-e5-small を Core ML (.mlpackage) へ変換する。

平均プーリングと L2 正規化をモデル側へ焼き込み、Swift 側は
「トークン列 → 正規化済み 384 次元ベクトル」だけを扱えばよい状態にする。

使い方:
    python convert_model.py
出力:
    output/MultilingualE5Small.mlpackage  (FP16, 可変長入力 1〜512)
    output/tokenizer/                     (tokenizer.json ほか、Phase 3 でアプリに同梱)
"""

from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "intfloat/multilingual-e5-small"
MAX_SEQ_LEN = 512
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PACKAGE_PATH = OUTPUT_DIR / "MultilingualE5Small.mlpackage"


class E5Embedder(torch.nn.Module):
    """attention mask を考慮した平均プーリング + L2 正規化を含むラッパー。

    なぜ焼き込むか: Swift 側でプーリングを再実装すると mask の扱いを
    間違えやすく、Python 実装との一致検証も難しくなる。モデルの出力を
    「そのまま保存・比較できるベクトル」にしておけば、アプリ側の実装は
    最小になり、検証はコサイン類似度の比較だけで済む。
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # [B, L, 384]
        # パディング位置を除外した平均(e5 公式実装の average_pool と等価)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        # 保存時に正規化済みなら、検索時は「内積 = コサイン類似度」として扱える
        return torch.nn.functional.normalize(pooled, p=2, dim=1)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"[1/4] モデルとトークナイザを取得: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()

    print("[2/4] TorchScript へトレース")
    wrapper = E5Embedder(model).eval()
    example = tokenizer("passage: 変換の動作確認に使う自作の文。", return_tensors="pt")
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper, (example["input_ids"], example["attention_mask"])
        )

    print("[3/4] Core ML へ変換(FP16 / 可変長 1〜512)")
    # なぜ可変長か: 検索クエリは通常 10〜30 トークン程度で、固定 512 に
    # パディングすると注意計算 O(L^2) の無駄が大きい。短文はその長さ分の
    # 計算で済ませる。速度は Phase 3/6 で実測し、必要なら固定長へ切り替える。
    seq_len = ct.RangeDim(lower_bound=1, upper_bound=MAX_SEQ_LEN, default=128)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, seq_len), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, seq_len), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="embedding", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    mlmodel.short_description = (
        "multilingual-e5-small (mean pooling + L2 normalize 焼き込み済み, FP16)"
    )
    mlmodel.input_description["input_ids"] = "トークン ID 列 (1, 1..512)"
    mlmodel.input_description["attention_mask"] = "1=実トークン, 0=パディング"
    mlmodel.output_description["embedding"] = "L2 正規化済み 384 次元ベクトル"

    print(f"[4/4] 保存: {PACKAGE_PATH}")
    mlmodel.save(str(PACKAGE_PATH))
    tokenizer.save_pretrained(OUTPUT_DIR / "tokenizer")

    size_mb = sum(f.stat().st_size for f in PACKAGE_PATH.rglob("*") if f.is_file()) / 1e6
    print(f"完了: {PACKAGE_PATH.name} ({size_mb:.1f} MB)")
    print("次は validate_model.py で PyTorch との一致を検証してください。")


if __name__ == "__main__":
    main()
