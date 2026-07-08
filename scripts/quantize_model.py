#!/usr/bin/env python3
"""FP16 の Core ML モデルから INT8(weight-only 線形量子化)版を作り、品質を検証する。

なぜ weight-only か: 活性値まで量子化する方式より安全で、サイズ削減効果
(重みが支配的)はほぼ同じ。推論時は重みを FP16 へ戻して計算するため、
精度低下は重みの丸め誤差だけに限定される。

使い方:
    python quantize_model.py   (convert_model.py の実行後に)
出力:
    output/MultilingualE5SmallInt8.mlpackage
"""

from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from convert_model import MODEL_ID, PACKAGE_PATH, E5Embedder
from validate_model import CASES, COSINE_THRESHOLD, cosine

INT8_PACKAGE_PATH = Path(__file__).resolve().parent / "output" / "MultilingualE5SmallInt8.mlpackage"


def directory_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main() -> None:
    if not PACKAGE_PATH.exists():
        raise SystemExit(f"{PACKAGE_PATH} がありません。先に convert_model.py を実行してください。")

    print("[1/3] FP16 モデルを読み込み、INT8 へ量子化")
    mlmodel = ct.models.MLModel(str(PACKAGE_PATH))
    config = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    )
    quantized = cto.linear_quantize_weights(mlmodel, config=config)
    quantized.short_description = "multilingual-e5-small (INT8 weight-only 量子化)"
    quantized.save(str(INT8_PACKAGE_PATH))

    fp16_mb = directory_size_mb(PACKAGE_PATH)
    int8_mb = directory_size_mb(INT8_PACKAGE_PATH)
    print(f"サイズ: FP16 {fp16_mb:.1f} MB → INT8 {int8_mb:.1f} MB ({int8_mb / fp16_mb:.0%})")

    print("[2/3] PyTorch(FP32)との一致を検証")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    wrapper = E5Embedder(AutoModel.from_pretrained(MODEL_ID)).eval()
    int8_model = ct.models.MLModel(str(INT8_PACKAGE_PATH), compute_units=ct.ComputeUnit.CPU_ONLY)

    print(f"{'case':24} {'cos(pt,int8)':>13}")
    worst = 1.0
    for case_id, text in CASES:
        encoded = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            reference = wrapper(encoded["input_ids"], encoded["attention_mask"]).numpy()
        predicted = int8_model.predict({
            "input_ids": encoded["input_ids"].numpy().astype(np.int32),
            "attention_mask": encoded["attention_mask"].numpy().astype(np.int32),
        })["embedding"]
        similarity = cosine(reference, predicted)
        worst = min(worst, similarity)
        print(f"{case_id:24} {similarity:>13.6f}")

    print("[3/3] 判定")
    print(f"最小コサイン類似度: {worst:.6f}(FP16 の合格基準は {COSINE_THRESHOLD})")
    print("INT8 の検索品質への影響は Swift 側のベンチマーク(recall/nDCG)で最終確認する。")


if __name__ == "__main__":
    main()
