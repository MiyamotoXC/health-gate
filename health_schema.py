# -*- coding: utf-8 -*-
"""
健康管理助手 · 场景/合规/脱敏 统一定义（纯标准库，无第三方依赖）
==================================================================
本模块是「健康管理助手」场景规则与输出契约的**唯一语义来源**，被
medical_data_processor.py（语料工程）、health_eval.py（效果评测）、
human_review.py（人工审核）共同引用，避免同一套规则散落多处。

业务场景（对应健康管理门诊的四类高频任务，而非通用问诊）：
    report    体检/检验报告解读
    indicator 指标与参考区间咨询
    followup  随访与复诊
    plan      干预方案与生活方式
    referral  转诊与就医决策

三条硬约束（医疗场景合规要求，不可绕过）：
    A. 脱敏优先    语料入库前先做 PHI 脱敏
    B. 表述合规    绝对化诊断 / 承诺疗效 / 具体处方剂量 一律剔除或降级
    C. 不确定转人工 信息不足的样本标记 need_review=True，进人工审核队列而非训练集
"""

import re


# ============================================================
# 一、人设与输出契约
# ============================================================


SYSTEM_PROMPT = """你是「健康管理助手」，服务于高端健康管理门诊，面向已完成体检/检验的客户提供报告解读、指标咨询、随访提醒与生活方式干预建议。

输出必须严格遵循以下四段结构，缺一段即视为不合格：
【结论】用一句话给出可执行的判断（正常 / 需关注 / 建议就医），不展开。
【依据】列出支撑结论的指标、参考区间或指南口径，每条都能追溯到具体数值或依据。
【建议】给出可执行的下一步（复查项目与间隔、生活方式调整、需监测的指标）。
【风险提示】说明哪些情况需要立即就医；信息不足以判断时必须写「需人工复核」，不得猜测。

四条红线：
1. 不做确诊性结论，不使用「你就是xx病」「肯定是」等绝对化表述。
2. 不给出具体处方剂量；涉及用药统一回答「请遵医嘱 / 需医生开具」。
3. 信息不足时主动说明需结合临床检查并由医生判断，禁止编造数值。
4. 不引用未标注的来源，不承诺疗效。"""

# 训练模板：{system} / {question} / {answer}
TRAIN_TEMPLATE = """{}

### 问题：
{}

### 回答：
{}"""

# 评测与生成共用的段落契约
OUTPUT_SECTIONS = ["【结论】", "【依据】", "【建议】", "【风险提示】"]


# ============================================================
# 二、场景分类
# ============================================================

SCENE_RULES = {
    "report": (
        "体检/检验报告解读",
        ["报告", "体检", "化验单", "检查单", "结果单", "体检单", "帮我看", "怎么看", "解读", "上面写"],
    ),
    "indicator": (
        "指标与参考区间咨询",
        ["指标", "参考值", "参考区间", "正常值", "偏高", "偏低", "箭头", "异常", "血常规",
         "尿常规", "血脂", "血糖", "血压", "转氨酶", "结节", "尿酸", "肌酐", "胆固醇"],
    ),
    "followup": (
        "随访与复诊",
        ["复查", "复诊", "随访", "多久再", "什么时候再", "隔多久", "再次检查", "下一次", "三个月后"],
    ),
    "plan": (
        "干预方案与生活方式",
        ["怎么办", "注意什么", "吃什么", "饮食", "运动", "作息", "减肥", "戒烟", "忌口",
         "调理", "保养", "预防", "改善"],
    ),
    "referral": (
        "转诊与就医决策",
        ["要不要去医院", "需要就医", "挂什么科", "看什么科", "严重吗", "要不要紧", "急诊", "进一步检查"],
    ),
}

DEFAULT_SCENE = "indicator"


def classify_scene(question):
    """按关键词打分给问题打场景标签。返回 (scene_code, scene_name, 命中数)。"""
    text = question or ""
    best, best_hits = DEFAULT_SCENE, 0
    for code, (name, keywords) in SCENE_RULES.items():
        hits = sum(1 for k in keywords if k in text)
        if hits > best_hits:
            best, best_hits = code, hits
    return best, SCENE_RULES[best][0], best_hits


# ============================================================
# 三、PHI 脱敏（约束 A）
# ============================================================

PHI_PATTERNS = [
    (r"\d{17}[\dXx]", "[身份证]"),
    (r"\b1[3-9]\d{9}\b", "[手机号]"),
    (r"(?:住院号|病案号|就诊号|门诊号|体检号)[:：]?\s*\d+", "[就医编号]"),
    (r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?", "[日期]"),
    (r"(?:我叫|本人|患者)\s*[\u4e00-\u9fa5]{2,4}(?=[，,。；;\s]|$)", "[姓名]"),
]


def desensitize(text):
    """对单条文本做 PHI 脱敏。顺序敏感：长模式先于短模式。"""
    for pattern, replacement in PHI_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text


# ============================================================
# 四、表述合规（约束 B）
# ============================================================

RISK_PATTERNS = {
    "绝对化诊断": [
        r"你(?:就是|肯定是|一定是|患了|得了)[^。！？]{0,12}(?:病|症|炎|癌|瘤)",
        r"(?:确诊|百分百|绝对是|肯定是)[^。！？]{0,12}",
    ],
    "承诺疗效": [
        r"(?:保证|一定|百分百|彻底)(?:能|会)?(?:治好|治愈|康复|根治)",
        r"药到病除",
        r"完全治愈",
    ],
    "处方剂量": [
        r"每次\s*\d+\s*(?:片|粒|mg|毫升|ml|克|g)",
        r"(?:一天|每日)\s*[一二三四1234]\s*次[，,]?\s*每次",
        r"口服\s*\d+\s*(?:mg|g|片|粒)",
    ],
}


def scan_risk(text):
    """返回命中的风险类别列表（已排序）。"""
    text = text or ""
    return sorted(
        category
        for category, patterns in RISK_PATTERNS.items()
        if any(re.search(p, text) for p in patterns)
    )


def check_compliance(answer):
    """合规检查。返回 (ok: bool, reason: str)。ok=False 表示不得直接进训练集。"""
    risks = scan_risk(answer)
    if risks:
        return False, "表述不合规:" + "/".join(risks)
    return True, "合规"


# ============================================================
# 五、低置信度判定（约束 C）
# ============================================================

LOW_CONF_THRESHOLD = 0.15      # BLEU-2 低于该值视为低置信度
MIN_ADVICE_LEN = 30            # 回答少于该字数视为信息不足

ACTION_PATTERN = r"(建议|需要|应该|可以|如|若|请)"


def needs_human_review(question, answer, scene_hits=0):
    """判定样本是否需要转人工。返回 (need: bool, reason: str)。"""
    answer = answer or ""
    if len(answer) < MIN_ADVICE_LEN:
        return True, "回答过短"
    if scene_hits == 0:
        return True, "场景未命中"
    if not re.search(ACTION_PATTERN, answer):
        return True, "缺少可执行建议"
    return False, ""


# ============================================================
# 六、输出结构校验（评测与审核共用）
# ============================================================

REFUSAL_PATTERNS = [
    r"^\s*(?:抱歉|对不起|很抱歉)",
    r"我(?:无法|不能|不具备)",
    r"作为(?:一个)?(?:人工智能|AI|语言模型)",
    r"建议咨询(?:专业)?医生$",
]

REFERRAL_PATTERNS = [r"就医", r"去医院", r"挂.{0,4}科", r"看医生", r"门诊", r"急诊", r"请医生"]


def structure_compliance(text):
    """校验四段式输出。返回 (ok: bool, 缺失段落列表)。"""
    text = text or ""
    missing = [s for s in OUTPUT_SECTIONS if s not in text]
    return (not missing), missing


def is_refusal(text):
    """是否属于无内容拒答（拒答本身也应被计入异常，需人工补写）。"""
    text = (text or "").strip()
    return any(re.search(p, text, re.MULTILINE) for p in REFUSAL_PATTERNS)


def has_referral(text):
    """是否包含就医/转诊提示。"""
    text = text or ""
    return any(re.search(p, text) for p in REFERRAL_PATTERNS)
