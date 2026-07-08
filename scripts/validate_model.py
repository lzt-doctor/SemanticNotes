#!/usr/bin/env python3
"""変換した Core ML モデルを PyTorch 実装と突き合わせて検証する。

検証内容:
  1. 日英 × "query: "/"passage: " の8文で、PyTorch(FP32)と Core ML(FP16)の
     出力ベクトルのコサイン類似度が 0.999 を超えること(PLAN の完了条件)。
  2. パディング不変性: 同じ文をパディング付きで入力しても出力が変わらないこと
     (焼き込んだ平均プーリングが attention mask を正しく使っている証拠)。
  3. Phase 3 の Swift テストで使う「入力文・トークン列・参照ベクトル」を
     ../SemanticNotesTests/Resources/ReferenceEmbeddings.json へ書き出す。

使い方:
    python validate_model.py   (convert_model.py の実行後に)
"""

import json
from datetime import date
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from convert_model import MODEL_ID, PACKAGE_PATH, E5Embedder

COSINE_THRESHOLD = 0.999
REFERENCE_JSON = (
    Path(__file__).resolve().parent.parent
    / "SemanticNotesTests" / "Resources" / "ReferenceEmbeddings.json"
)

# 検証文はすべて自作(CLAUDE.md の制約: 既存文章の流用禁止)。
# E5 系の必須接頭辞 "query: " / "passage: " を日英両方でカバーする。
CASES = [
    ("ja-query-budget", "query: 先週の会議で決まった予算の担当者は誰?"),
    ("ja-passage-budget", "passage: 予算の見直しは経理の田中さんが担当することになった。締め切りは金曜日。"),
    ("ja-query-trip", "query: 旅行の持ち物リスト"),
    ("ja-passage-trip", "passage: 沖縄旅行の持ち物: 日焼け止め、水着、モバイルバッテリー、折りたたみ傘。"),
    ("en-query-index", "query: where did I write the steps to rebuild the search index?"),
    ("en-passage-index", "passage: To rebuild the search index, remove stale chunks first and then re-embed every note."),
    ("en-query-swift", "query: notes about swift concurrency"),
    ("mixed-passage-swift", "passage: Swift の並行処理メモ: actor は状態を隔離し、Sendable は境界を越える値の安全性を示す。"),
]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    if not PACKAGE_PATH.exists():
        raise SystemExit(f"{PACKAGE_PATH} がありません。先に convert_model.py を実行してください。")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    wrapper = E5Embedder(AutoModel.from_pretrained(MODEL_ID)).eval()
    # なぜ CPU_ONLY: 検証の再現性のため。ANE/GPU は環境で数値が揺れることがある。
    mlmodel = ct.models.MLModel(str(PACKAGE_PATH), compute_units=ct.ComputeUnit.CPU_ONLY)

    results = []
    reference_cases = []
    for case_id, text in CASES:
        encoded = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            torch_vec = wrapper(encoded["input_ids"], encoded["attention_mask"]).numpy()

        coreml_out = mlmodel.predict({
            "input_ids": encoded["input_ids"].numpy().astype(np.int32),
            "attention_mask": encoded["attention_mask"].numpy().astype(np.int32),
        })["embedding"]

        similarity = cosine(torch_vec, coreml_out)
        norm = float(np.linalg.norm(coreml_out))
        results.append((case_id, similarity, norm, len(encoded["input_ids"][0])))
        reference_cases.append({
            "id": case_id,
            "text": text,
            "token_ids": encoded["input_ids"][0].tolist(),
            # 参照は FP32 の PyTorch 出力(真値側)。小数6桁で十分(コサイン比較のため)
            "embedding": [round(x, 6) for x in torch_vec.flatten().tolist()],
        })

    # パディング不変性: 同じ文を 64 トークンまでパディングしても出力が同じこと
    base = tokenizer(CASES[0][1], return_tensors="pt")
    padded = tokenizer(CASES[0][1], return_tensors="pt", padding="max_length", max_length=64)
    padded_out = mlmodel.predict({
        "input_ids": padded["input_ids"].numpy().astype(np.int32),
        "attention_mask": padded["attention_mask"].numpy().astype(np.int32),
    })["embedding"]
    base_out = mlmodel.predict({
        "input_ids": base["input_ids"].numpy().astype(np.int32),
        "attention_mask": base["attention_mask"].numpy().astype(np.int32),
    })["embedding"]
    padding_similarity = cosine(base_out, padded_out)

    print(f"{'case':24} {'tokens':>6} {'cos(pt,cml)':>12} {'|v|':>8}")
    for case_id, similarity, norm, n_tokens in results:
        print(f"{case_id:24} {n_tokens:>6} {similarity:>12.6f} {norm:>8.5f}")
    print(f"{'padding-invariance':24} {'':>6} {padding_similarity:>12.6f}")

    min_similarity = min(s for _, s, _, _ in results)
    failed = min_similarity <= COSINE_THRESHOLD or padding_similarity <= COSINE_THRESHOLD
    if failed:
        raise SystemExit(f"NG: コサイン類似度が閾値 {COSINE_THRESHOLD} を下回りました")

    REFERENCE_JSON.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_JSON.write_text(json.dumps({
        "model_id": MODEL_ID,
        "dimension": 384,
        "normalized": True,
        "generated_at": date.today().isoformat(),
        "validation": {
            "min_cosine_vs_coreml_fp16": round(min_similarity, 6),
            "padding_invariance_cosine": round(padding_similarity, 6),
            "compute_units": "CPU_ONLY",
        },
        "cases": reference_cases,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\nOK: 最小コサイン類似度 {min_similarity:.6f} > {COSINE_THRESHOLD}")
    print(f"参照ベクトルを書き出しました: {REFERENCE_JSON}")


if __name__ == "__main__":
    main()
