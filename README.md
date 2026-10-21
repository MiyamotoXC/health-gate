# 健康管理助手 · 语料与效果安全闸门

> 一个**面向健康管理门诊（体检报告解读 / 指标咨询 / 随访 / 干预方案 / 转诊决策）**的医疗 AI 语料工程 + 效果评测 + 人工审核闭环。
> 核心价值不是"又一个医疗问答模型"，而是一套**让医疗 AI 能安全上线前的闸门**：脱敏、合规、低置信度转人工、评测、审计留痕。
> 作者：[@MiyamotoXC](https://github.com/MiyamotoXC) ｜ 许可证：MIT ｜ 觉得有用请 Star / Fork 二次开发。

[![CI](https://github.com/MiyamotoXC/health-gate/actions/workflows/ci.yml/badge.svg)](https://github.com/MiyamotoXC/health-gate/actions/workflows/ci.yml)
> ↑ 构建状态由 GitHub Actions 自动运行（clone 后 `python demo.py` 即触发端到端校验）。

---

## 一、它解决什么问题

通用医疗问答 SFT 语料有两个工程上致命的盲区：

1. **数据带 PHI、表述不合规**——直接进训练集是数据安全事故。
2. **只靠 BLEU 评测**——BLEU 衡量"像不像参考答案"，衡量不了"能不能上线"。

本项目用一套轻量规则把这两个盲区堵上，且**不依赖任何大模型**：

| 能力 | 模块 | 说明 |
|---|---|---|
| 场景分类 | `health_schema.classify_scene` | 5 类高频任务（report/indicator/followup/plan/referral）自动打标 |
| PHI 脱敏 | `health_schema.desensitize` | 身份证/手机/就医编号/日期/姓名 → 占位符 |
| 表述合规 | `health_schema.check_compliance` | 绝对化诊断 / 承诺疗效 / 处方剂量 → 标记需复核，剔除出训练集 |
| 低置信度转人工 | `health_schema.needs_human_review` | 回答过短 / 场景未命中 / 缺可执行建议 → 标记需复核 |
| 效果评测 | `health_eval.py` | BLEU-2 + 结构合规率 / 风险命中率 / 转诊覆盖率 / 拒答率 / 低置信度率，按场景分组，seed 可复现 |
| 人工审核闭环 | `human_review.py` | 待审队列导出 → 人工改写 → 稳定 id 覆盖写回 → 差异对照 → 审计留痕 |

---

## 二、目录结构

```
健康管理助手/
├── health_schema.py          # 场景/合规/脱敏 规则的唯一定义处（纯标准库，无三方依赖）
├── health_eval.py            # 场景化效果评测
├── human_review.py           # 人工审核层
├── medical_data_processor.py # 真实语料 Pipeline（6 科室 CSV → 清洗 → 场景标注 → 待审队列）
├── demo.py                   # 端到端演示（零三方依赖，任何环境可跑）
├── examples/
│   └── sample_records.jsonl  # 样例记录（含 PHI / 风险表述 / 过短回答）
├── demo_output/              # demo 运行产物（评测报告 / 审核队列 / 审计日志）
├── requirements.txt
└── README.md
```

---

## 三、快速开始（任何人，无需 GPU）

```bash
python demo.py                          # 内置 4 条样例，跑完三部分
python demo.py --data examples/sample_records.jsonl
python demo.py --quick                  # 只看质检前置（脱敏/合规/转人工）
python health_eval.py --demo            # 单独看评测
python health_eval.py --pred P.jsonl --ref R.jsonl   # 自有模型输出评测
python human_review.py --build --records data.jsonl --out queue.jsonl
# 人工编辑 queue.jsonl 的 answer / reviewer / note
python human_review.py --apply --records data.jsonl --overrides queue.jsonl --out merged.jsonl
```

`demo.py` 会依次展示：

1. **质检前置**：每条样本的场景标签、脱敏后问题、是否合规、是否转人工。
2. **效果评测**：整体 + 分场景的六项指标。
3. **审核闭环**：把待审样本改写成合规四段式 → 覆盖写回 → 审计留痕，并验证覆盖后"仍需人工"归零。

所有产物落在 `demo_output/`。

---

## 四、输出契约（模型训练目标结构）

```
【结论】用一句话给出可执行的判断（正常 / 需关注 / 建议就医），不展开。
【依据】列出支撑结论的指标、参考区间或指南口径，每条都能追溯到具体数值。
【建议】给出可执行的下一步（复查项目与间隔、生活方式调整、需监测的指标）。
【风险提示】说明哪些情况需要立即就医；信息不足时必须写「需人工复核」，不得猜测。
```

四条红线：不做确诊性结论 / 不给具体处方剂量 / 信息不足主动说需医生判断 / 不承诺疗效。

---

## 五、与医疗 AI 岗位 JD 的对应关系

| JD 要求 | 本项目证据 |
|---|---|
| 报告解读 / 随访 / 干预方案 / 辅助决策 | 五类场景分类 + 四段式输出契约 |
| AI 效果评测（基线 / 低置信度 / 异常提醒） | `health_eval.py` 六项指标 + seed 可复现抽样 + 低置信度裁决 |
| 人工审核机制 | `human_review.py` 待审队列 → 覆盖写回 → 差异对照 → 审计留痕 |
| 数据安全 / 脱敏 / 访问留痕 | `desensitize()` + 审计日志（旧值/新值/审核人/时间） |
| Prompt 工程 | `SYSTEM_PROMPT` 四条红线 + 固定四段结构，可控输出 |
| 数据工程 → 训练 全链路 | `medical_data_processor.py` 可复现 |

---

## 六、诚实边界

- **未做 OCR / 体检报告结构化本体**：本项目处理的是**已结构化**的问答语料。报告 PDF/图片 → 字段这一步未落地；口径是"文档→结构化字段→人工审核覆盖"这条链已闭环，OCR 只是把解析器换成 PaddleOCR / PP-Structure。
- **未绑定具体大模型**：评测与审核对模型中立，接任意输出 JSONL 即可。
- **0.8B 基座 + LoRA 仅用于验证工程链路**，不代表生产效果。

---

## 七、简历可用条目（项目经历）

> 也可直接看仓库根目录的「简历项目经历」整合文档。下面这条可直接贴进简历。

**健康管理助手 · 医疗 AI 语料安全闸门** ｜ Python 标准库 / 规则引擎 / 场景化评测
- 面向健康管理门诊（报告解读/指标咨询/随访/干预/转诊），将通用医疗 SFT 语料改造为「安全上线闸门」：5 类场景自动打标、PHI 脱敏、合规过滤、低置信度转人工、场景化评测、人工审核审计闭环，纯标准库实现、不依赖大模型。
- 落地四段式输出契约（结论/依据/建议/风险提示）+ 四条红线；BLEU-2 + 6 项上线指标（结构合规率/风险命中率/转诊率/拒答率/低置信度率），seed 可复现；人工审核覆盖后「仍需人工」归零。
- 对齐 联影「长寿门诊/健康管理」全栈岗：报告解读、随访、干预方案、辅助决策、数据安全脱敏、人工审核机制。

## 八、License

代码以 MIT 发布；样例数据来自公开医疗问答语料，**已脱敏**，仅供教学演示，不构成任何医疗建议。
