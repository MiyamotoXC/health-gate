# -*- coding: utf-8 -*-
"""
健康管理助手 · 场景化效果评测
==================================================================
BLEU 只能衡量"像不像参考答案"，医疗场景还要衡量"能不能上线"。
本模块在 BLEU 之上补四类硬指标，并对每条样本给出要不要转人工的裁决。

指标口径：
    1. bleu2              与参考答案的 1-2gram 相似度（jieba 可用时用 jieba 实现，
                          否则降级为字符 2gram，报告里用 bleu_backend 标明）
    2. structure_rate     四段式结构（【结论】【依据】【建议】【风险提示】）合规率
    3. risk_hit_rate      风险表述命中率（绝对化诊断 / 承诺疗效 / 处方剂量）——越低越好
    4. referral_rate      就医/转诊提示覆盖率
    5. refusal_rate       无内容拒答率
    6. need_review_rate   低置信度率（BLEU 低于阈值 / 结构缺失 / 命中风险 / 拒答，任一命中即计）

裁决：need_review=True 的样本不进线上，导出到人工审核队列（见 human_review.py）。

用法：
    python health_eval.py --pred processed_data/predictions.jsonl \
                          --ref  processed_data/medical_alpaca_val.jsonl \
                          --out  processed_data/health_eval_report.json
    python health_eval.py --demo            # 内置样例，零依赖自测
"""

import argparse
import json
import os
import random
import re


# ============================================================
# 依赖：BLEU 后端（缺 jieba 时降级，保证脚本永远能跑）
# ============================================================

try:
    from bleu_evaluation import calculate_bleu_simple as _bleu_jieba
    BLEU_BACKEND = "jieba(bleu_evaluation)"
except Exception:                                    # pragma: no cover
    _bleu_jieba = None
    BLEU_BACKEND = "char2gram(fallback)"


def _char_ngrams(text, n):
    text = re.sub(r"\s+", "", text or "")
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def bleu2(reference, candidate):
    """统一的 BLEU-2 入口，返回 float。"""
    if _bleu_jieba is not None:
        return float(_bleu_jieba(reference, candidate)["bleu_score"])

    import math
    from collections import Counter
    ref1, cand1 = _char_ngrams(reference, 1), _char_ngrams(candidate, 1)
    ref2, cand2 = _char_ngrams(reference, 2), _char_ngrams(candidate, 2)
    if not cand1 or not cand2:
        return 0.0

    def precision(ref_g, cand_g):
        rc, cc = Counter(ref_g), Counter(cand_g)
        hit = sum(min(v, rc.get(k, 0)) for k, v in cc.items())
        return (hit + 1) / (sum(cc.values()) + 1)     # +1 平滑

    p1, p2 = precision(ref1, cand1), precision(ref2, cand2)
    bp = 1.0 if len(cand1) >= len(ref1) else math.exp(1 - len(ref1) / max(len(cand1), 1))
    return round(bp * math.sqrt(max(p1, 1e-10) * max(p2, 1e-10)), 4)


# ============================================================
# 单条评测
# ============================================================

def evaluate_one(question, reference, prediction, scene=None):
    """
    对单条 (问题, 参考答案, 模型输出) 打分并裁决是否转人工。
    scene 为空时用 health_schema.classify_scene 自动补。
    """
    from health_schema import (
        classify_scene, scan_risk, structure_compliance,
        is_refusal, has_referral, LOW_CONF_THRESHOLD,
    )

    if not scene:
        scene, scene_name, _ = classify_scene(question)
    else:
        scene_name = ""

    score = bleu2(reference, prediction)
    struct_ok, missing = structure_compliance(prediction)
    risks = scan_risk(prediction)
    refusal = is_refusal(prediction)
    referral = has_referral(prediction)
    low_conf = score < LOW_CONF_THRESHOLD

    reasons = []
    if low_conf:
        reasons.append("低置信度")
    if not struct_ok:
        reasons.append("结构缺失:" + "/".join(missing))
    if risks:
        reasons.append("风险表述:" + "/".join(risks))
    if refusal:
        reasons.append("拒答")

    return {
        "question": question,
        # 带上原文，人工审核队列才有可编辑的对象（human_review 依赖此字段）
        "prediction": prediction,
        "reference": reference,
        "scene": scene,
        "scene_name": scene_name,
        "bleu2": round(score, 4),
        "structure_ok": struct_ok,
        "missing_sections": missing,
        "risk_hits": risks,
        "has_referral": referral,
        "is_refusal": refusal,
        "need_review": bool(reasons),
        "review_reasons": reasons,
    }


# ============================================================
# 批量评测
# ============================================================

def _rate(items, key):
    """items 中 key 为真的占比。"""
    return round(sum(1 for x in items if x[key]) / len(items), 4) if items else 0.0


def _summarize(items):
    if not items:
        return {"count": 0}
    scores = sorted(x["bleu2"] for x in items)
    n = len(scores)
    return {
        "count": n,
        "bleu2_avg": round(sum(scores) / n, 4),
        "bleu2_p50": round(scores[n // 2], 4),
        "bleu2_min": round(scores[0], 4),
        "bleu2_max": round(scores[-1], 4),
        "structure_rate": _rate(items, "structure_ok"),
        "risk_hit_rate": _rate(items, "risk_hits"),
        "referral_rate": _rate(items, "has_referral"),
        "refusal_rate": _rate(items, "is_refusal"),
        "need_review_rate": _rate(items, "need_review"),
    }


def evaluate_batch(records, seed=20261012, sample_n=None):
    """
    records: [{question, reference, prediction, scene?}, ...]
    sample_n: 不为空时按 seed 随机抽样（可复现的回归评测）
    """
    pool = list(records)
    sampled = False
    if sample_n and 0 < sample_n < len(pool):
        rng = random.Random(seed)
        pool = rng.sample(pool, sample_n)
        sampled = True

    details = [evaluate_one(r["question"], r.get("reference", ""),
                            r.get("prediction", ""), r.get("scene")) for r in pool]

    by_scene = {}
    for d in details:
        by_scene.setdefault(d["scene_name"] or d["scene"], []).append(d)

    return {
        "meta": {
            "backend": BLEU_BACKEND,
            "seed": seed,
            "sampled": sampled,
            "sample_size": len(pool),
            "total_size": len(records),
        },
        "overall": _summarize(details),
        "by_scene": {k: _summarize(v) for k, v in sorted(by_scene.items())},
        "details": details,
    }


# ============================================================
# IO
# ============================================================

def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_json(data, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"报告已写出: {path}")


def print_report(report):
    o = report["overall"]
    print("\n" + "=" * 62)
    print("  健康管理助手 · 场景化评测报告")
    print("=" * 62)
    print(f"  样本数 {report['meta']['sample_size']}/{report['meta']['total_size']}"
          f"   抽样={report['meta']['sampled']}   seed={report['meta']['seed']}"
          f"   BLEU后端={report['meta']['backend']}")
    print("-" * 62)
    print(f"  BLEU-2 均值        {o.get('bleu2_avg')}")
    print(f"  四段结构合规率     {o.get('structure_rate')}")
    print(f"  风险表述命中率     {o.get('risk_hit_rate')}   (越低越好)")
    print(f"  转诊提示覆盖率     {o.get('referral_rate')}")
    print(f"  拒答率             {o.get('refusal_rate')}")
    print(f"  低置信度(转人工)率 {o.get('need_review_rate')}")
    print("-" * 62)
    for scene, s in report["by_scene"].items():
        print(f"  [{scene}] n={s['count']}  bleu={s['bleu2_avg']}  "
              f"结构={s['structure_rate']}  转人工={s['need_review_rate']}")
    print("=" * 62)


# ============================================================
# 主流程
# ============================================================

DEMO_RECORDS = [
    {
        "question": "体检报告上写总胆固醇 6.3，参考区间 2.8-5.17，这个要紧吗？",
        "reference": "属于偏高。建议 3 个月后复查血脂四项，同时控制饱和脂肪摄入、增加有氧运动。",
        "prediction": "【结论】总胆固醇偏高，需关注但暂不需用药。\n【依据】6.3 mmol/L 高于参考区间上限 5.17。\n"
                      "【建议】3 个月后复查血脂四项；减少动物脂肪，每周有氧运动 150 分钟。\n"
                      "【风险提示】若合并高血压、糖尿病或有家族史，请尽早就医评估。",
    },
    {
        "question": "我最近总是头晕怎么办？",
        "reference": "需结合血压、血常规与颈椎检查判断，建议先测量血压并就医明确原因。",
        "prediction": "你就是贫血，肯定是的，多吃点补血的就行。",
    },
    {
        "question": "肺小结节 4mm，需要复查吗？什么时候再检查？",
        "reference": "4mm 属低风险，按指南建议 6-12 个月复查低剂量 CT，观察是否增大。",
        "prediction": "抱歉，我无法提供医疗建议。",
    },
]


def main():
    ap = argparse.ArgumentParser(description="健康管理助手 · 场景化效果评测")
    ap.add_argument("--pred", help="模型输出 JSONL：{question, prediction(或 output)}")
    ap.add_argument("--ref", help="参考答案 JSONL：{input/question, output/reference}")
    ap.add_argument("--out", default="processed_data/health_eval_report.json")
    ap.add_argument("--sample", type=int, default=0, help="随机抽样条数（0=全量）")
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--demo", action="store_true", help="用内置样例自测")
    args = ap.parse_args()

    if args.demo:
        records = DEMO_RECORDS
    else:
        if not (args.pred and args.ref):
            print("需要同时给 --pred 和 --ref，或用 --demo 自测")
            return
        preds, refs = load_jsonl(args.pred), load_jsonl(args.ref)
        if len(preds) != len(refs):
            print(f"[警告] 条数不一致 pred={len(preds)} ref={len(refs)}，按较短的一方对齐")
        records = []
        for p, r in zip(preds, refs):
            records.append({
                "question": p.get("question") or r.get("input") or r.get("question", ""),
                "reference": r.get("reference") or r.get("output") or r.get("answer", ""),
                "prediction": p.get("prediction") or p.get("output") or p.get("answer", ""),
                "scene": p.get("scene") or r.get("scene"),
            })

    report = evaluate_batch(records, seed=args.seed, sample_n=args.sample or None)
    print_report(report)
    save_json(report, args.out)


if __name__ == "__main__":
    main()
