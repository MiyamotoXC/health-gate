# -*- coding: utf-8 -*-
"""
健康管理助手 · 人工审核层
==================================================================
医疗场景里"模型说得不对"比"模型没说"更严重，所以低置信度样本不能自动上线。
本模块把「待审队列导出 → 人工改写 → 覆盖写回 → 审计留痕」做成可复现的闭环：

    Step 1  build_queue()      从清洗/评测结果里挑出 need_review=True 的样本 -> review_queue.jsonl
    Step 2  （人工）            审核人直接编辑 review_queue.jsonl 里的 answer 字段
    Step 3  apply_overrides()  按稳定 id 覆盖回主数据集 -> merged.jsonl + audit_log.jsonl

稳定 id：对 question 做 md5 前 12 位，保证同一问题在多轮迭代中 id 不变，
        覆盖关系不会因顺序变化而错位。

审计留痕：每条覆盖都记录 id / 审核人 / 时间 / 旧值 / 新值，
        满足健康数据「访问留痕、可追溯」的合规要求。

用法：
    python human_review.py --build --records processed_data/medical_alpaca_full.jsonl \
                           --out processed_data/review_queue.jsonl
    python human_review.py --apply --records processed_data/medical_alpaca_full.jsonl \
                           --overrides processed_data/review_overrides.json \
                           --out processed_data/medical_alpaca_merged.jsonl
"""

import argparse
import hashlib
import json
import os
from datetime import datetime


def record_id(question):
    """对问题文本做稳定哈希，作为覆盖关系的键。"""
    return hashlib.md5((question or "").strip().encode("utf-8")).hexdigest()[:12]


def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_jsonl(data, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"已写出 {len(data)} 条 -> {path}")


def save_json(data, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已写出 -> {path}")


# ============================================================
# Step 1: 导出待审队列
# ============================================================

def build_queue(records, out_path=None):
    """
    挑出 need_review=True 的样本，附加稳定 id 与待审原因，导出为人工可编辑的 JSONL。
    records 既支持清洗产出的 {question, answer, ...}，
    也支持评测产出的 {question, prediction, ...}（后者把 prediction 命名为 answer）。
    """
    queue = []
    for r in records:
        if not r.get("need_review"):
            continue
        answer = r.get("answer") or r.get("prediction") or r.get("output") or ""
        queue.append({
            "id": record_id(r.get("question") or r.get("input") or ""),
            "question": r.get("question") or r.get("input") or "",
            "answer": answer,
            "scene": r.get("scene") or r.get("scene_name") or "",
            "review_reasons": r.get("review_reasons") or (
                [r["review_reason"]] if r.get("review_reason") else []),
            "reviewer": "",
            "note": "",
        })

    if out_path:
        save_jsonl(queue, out_path)
    print(f"待审队列: {len(queue)} / {len(records)} 条")
    return queue


# ============================================================
# Step 2/3: 覆盖写回 + 审计留痕
# ============================================================

def load_overrides(path):
    """
    读取人工改写结果。支持两种形态：
      A) {"<id>": {"answer": "...", "reviewer": "...", "note": "..."}}
      B) 直接传 review_queue.jsonl（人工就地编辑 answer 后的版本）
    """
    if path.lower().endswith(".jsonl"):
        rows = load_jsonl(path)
        overrides = {}
        for r in rows:
            if r.get("answer") and r.get("id"):
                overrides[r["id"]] = {
                    "answer": r["answer"],
                    "reviewer": r.get("reviewer", "unknown"),
                    "note": r.get("note", ""),
                }
        return overrides

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_overrides(records, overrides, reviewer="unknown", out_path=None,
                    audit_path=None):
    """
    按稳定 id 把人工改写覆盖回主数据集。
    返回 (merged_records, audit_log)；被覆盖的样本自动解除 need_review 标记。
    """
    merged, audit = [], []
    ts = datetime.now().isoformat(timespec="seconds")

    for r in records:
        rid = record_id(r.get("question") or r.get("input") or "")
        ov = overrides.get(rid)
        new = dict(r)

        if ov and ov.get("answer"):
            old_answer = r.get("answer") or r.get("prediction") or r.get("output") or ""
            if old_answer != ov["answer"]:
                audit.append({
                    "id": rid,
                    "timestamp": ts,
                    "reviewer": ov.get("reviewer") or reviewer,
                    "old": old_answer,
                    "new": ov["answer"],
                    "note": ov.get("note", ""),
                })
                if "answer" in new:
                    new["answer"] = ov["answer"]
                elif "prediction" in new:
                    new["prediction"] = ov["answer"]
                else:
                    new["output"] = ov["answer"]
                new["need_review"] = False
                new["review_reason"] = ""
                new["review_reasons"] = []
                new["human_reviewed"] = True
        merged.append(new)

    if out_path:
        save_jsonl(merged, out_path)
    if audit_path:
        save_json(audit, audit_path)

    print(f"覆盖生效 {len(audit)} 条 / 主数据集 {len(records)} 条")
    return merged, audit


def preview_diff(audit, limit=5):
    """打印差异对照：逐处列出 旧值 -> 新值，供人工最后确认。"""
    print("\n" + "=" * 62)
    print("  人工审核差异对照")
    print("=" * 62)
    for i, a in enumerate(audit[:limit], 1):
        print(f"\n[{i}] id={a['id']}  审核人={a['reviewer']}  时间={a['timestamp']}")
        print(f"  旧值: {a['old'][:120]}")
        print(f"  新值: {a['new'][:120]}")
        if a.get("note"):
            print(f"  备注: {a['note']}")
    if len(audit) > limit:
        print(f"\n  …… 其余 {len(audit) - limit} 条见 audit_log")
    print("=" * 62)


# ============================================================
# 主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="健康管理助手 · 人工审核层")
    ap.add_argument("--build", action="store_true", help="导出待审队列")
    ap.add_argument("--apply", action="store_true", help="把人工改写覆盖回主数据集")
    ap.add_argument("--records", required=True, help="主数据集 JSONL")
    ap.add_argument("--out", default="processed_data/review_queue.jsonl")
    ap.add_argument("--overrides", help="人工改写结果（.json 或就地编辑后的 .jsonl）")
    ap.add_argument("--audit", default="processed_data/audit_log.json")
    ap.add_argument("--reviewer", default="unknown")
    args = ap.parse_args()

    records = load_jsonl(args.records)

    if args.build:
        build_queue(records, args.out)
        print("\n下一步：人工编辑上面的队列文件（改 answer、填 reviewer/note），然后：")
        print(f"  python human_review.py --apply --records {args.records} "
              f"--overrides {args.out}")
        return

    if args.apply:
        if not args.overrides:
            print("--apply 需要指定 --overrides")
            return
        overrides = load_overrides(args.overrides)
        _, audit = apply_overrides(records, overrides, args.reviewer,
                                   args.out, args.audit)
        preview_diff(audit)
        return

    print("需要指定 --build 或 --apply")


if __name__ == "__main__":
    main()
