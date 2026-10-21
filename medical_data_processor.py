# -*- coding: utf-8 -*-
"""
健康管理门诊语料工程：医疗对话数据的收集、清洗与训练就绪转换 Pipeline。

从 6 个科室的 CSV 中收集、清洗、合并医疗对话，输出训练就绪的数据集。

五类高频任务语料：
    report    体检/检验报告解读      indicator 指标与参考区间咨询
    followup  随访与复诊            plan      干预方案与生活方式
    referral  转诊与就医决策

在清洗流程中内置三层安全处理（默认开启）：
    A. PHI 脱敏      desensitize()    身份证/手机/就医编号/日期/姓名 -> 占位符
    B. 表述合规      check_compliance() 绝对化诊断/承诺疗效/处方剂量 -> 剔除出训练集
    C. 不确定转人工  needs_human_review() 信息不足 -> need_review=True，进人工审核队列

场景/合规/脱敏规则的唯一定义处是 health_schema.py，本模块只做引用与编排。
"""

import os
import re
import json
import hashlib
import pandas as pd
from collections import Counter, defaultdict
from datasets import Dataset

from health_schema import (
    SYSTEM_PROMPT,
    desensitize,
    classify_scene,
    scan_risk,
    check_compliance,
    needs_human_review,
)


# ========================================
# 第一步：数据收集 - 多源异构数据整合
# ========================================

# 科室映射表
DEPARTMENTS = {
    'IM_内科': '内科',
    'Surgical_外科': '外科',
    'Pediatric_儿科': '儿科',
    'Oncology_肿瘤科': '肿瘤科',
    'OAGD_妇产科': '妇产科',
    'Andriatria_男科': '男科'
}


def read_csv_with_encoding(file_path):
    """
    尝试使用不同编码读取CSV文件
    gb18030 是 GBK/GB2312 的超集，能处理更多字符
    如果严格模式全部失败，使用容错模式（替换无法解码的字节）
    """
    import io

    # 优先尝试严格模式（gb18030是GBK的超集，放在最前面）
    encodings = ['gb18030', 'utf-8', 'gbk', 'gb2312']
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"  编码: {encoding}, 行数: {len(df)}")
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            continue

    # 严格模式全部失败，使用gb18030容错模式（替换无法解码的字节）
    print(f"  使用gb18030容错模式读取")
    with open(file_path, 'r', encoding='gb18030', errors='replace') as f:
        content = f.read()
    df = pd.read_csv(io.StringIO(content))
    print(f"  容错模式读取成功, 行数: {len(df)}")
    return df


def collect_medical_data(data_dir):
    """
    从多个科室目录收集原始医疗数据
    返回: 原始DataFrame列表和统计信息
    """
    raw_data = []
    stats = defaultdict(int)
    
    for dept_dir, dept_name in DEPARTMENTS.items():
        dept_path = os.path.join(data_dir, dept_dir)
        if not os.path.exists(dept_path):
            print(f"[警告] 目录不存在: {dept_path}")
            continue
        
        print(f"\n--- 处理{dept_name}数据 ---")
        csv_files = [f for f in os.listdir(dept_path) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            file_path = os.path.join(dept_path, csv_file)
            print(f"正在读取: {csv_file}")
            
            try:
                df = read_csv_with_encoding(file_path)
                print(f"  列名: {df.columns.tolist()}")
                print(f"  行数: {len(df)}")
                
                df['department'] = dept_name
                df['source_file'] = csv_file
                raw_data.append(df)
                stats[dept_name] += len(df)
                
            except Exception as e:
                print(f"  [错误] 处理文件失败: {e}")
                continue
    
    print(f"\n========== 数据收集统计 ==========")
    for dept, count in stats.items():
        print(f"  {dept}: {count} 条")
    print(f"  总计: {sum(stats.values())} 条")
    
    return raw_data, stats


# ========================================
# 第二步：数据清洗
# ========================================

def extract_qa_fields(df):
    """
    从DataFrame中提取问题和回答字段
    支持多种列名格式
    """
    question_col = None
    answer_col = None
    
    question_candidates = ['question', 'ask', '问题', 'query', 'input']
    answer_candidates = ['answer', 'response', '回答', 'output', 'reply']
    
    for col in df.columns:
        col_lower = col.strip().lower()
        if col_lower in question_candidates and question_col is None:
            question_col = col
        if col_lower in answer_candidates and answer_col is None:
            answer_col = col
    
    return question_col, answer_col


def clean_text(text):
    """清洗单条文本"""
    if not isinstance(text, str):
        return ""
    
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    return text


def is_valid_qa(question, answer, min_q_len=5, max_q_len=500, min_a_len=10, max_a_len=2000):
    """
    验证问答对是否有效
    过滤条件：
    1. 非空
    2. 长度在合理范围内
    3. 不是纯标点或无意义内容
    """
    if not question or not answer:
        return False, "空值"
    
    if len(question) < min_q_len:
        return False, "问题过短"
    if len(question) > max_q_len:
        return False, "问题过长"
    if len(answer) < min_a_len:
        return False, "回答过短"
    if len(answer) > max_a_len:
        return False, "回答过长"
    
    if re.match(r'^[，。！？、；：""''（）\s]+$', question):
        return False, "问题仅含标点"
    
    meaningless_patterns = [
        r'^(你好|hello|hi|嗯|哦|好的|谢谢)$',
        r'^[\d\s]+$',
    ]
    for pattern in meaningless_patterns:
        if re.match(pattern, question, re.IGNORECASE):
            return False, "无意义问题"
    
    return True, "有效"


def deduplicate_by_question(data_list):
    """
    基于问题内容去重
    使用MD5哈希进行快速去重
    """
    seen_hashes = set()
    unique_data = []
    dup_count = 0
    
    for item in data_list:
        q_hash = hashlib.md5(item['question'].encode('utf-8')).hexdigest()
        if q_hash not in seen_hashes:
            seen_hashes.add(q_hash)
            unique_data.append(item)
        else:
            dup_count += 1
    
    print(f"  去重: 移除 {dup_count} 条重复数据")
    return unique_data


def clean_medical_data(raw_data_list, desensitize_phi=True, compliance_filter=True):
    """
    完整的数据清洗 Pipeline（面向健康管理门诊）
    输入: 原始DataFrame列表
    输出: 清洗后的数据列表，每条附带 scene / need_review / review_reason

    在原规则（空值 / 长度 / 无意义 / MD5去重）之上新增三层：
      desensitize_phi=True   -> 问答文本做 PHI 脱敏
      compliance_filter=True -> 命中绝对化诊断/承诺疗效/处方剂量的样本标记需复核，不进训练集
      （始终执行）           -> needs_human_review() 判定信息不足的样本标记需复核
    """
    print(f"\n========== 开始数据清洗 ==========")
    
    all_data = []
    filter_stats = Counter()
    
    for df in raw_data_list:
        q_col, a_col = extract_qa_fields(df)
        if q_col is None or a_col is None:
            print(f"  [警告] 无法识别问答列，跳过")
            continue
        
        for _, row in df.iterrows():
            question = clean_text(str(row[q_col]) if pd.notna(row[q_col]) else "")
            answer = clean_text(str(row[a_col]) if pd.notna(row[a_col]) else "")
            department = row.get('department', '未知')
            
            is_valid, reason = is_valid_qa(question, answer)
            if not is_valid:
                filter_stats[reason] += 1
                continue
            
            # --- 新增层 A：PHI 脱敏 ---
            if desensitize_phi:
                question = desensitize(question)
                answer = desensitize(answer)
            
            # --- 新增层：场景分类 ---
            scene, scene_name, scene_hits = classify_scene(question)
            
            # --- 新增层 B：表述合规 ---
            if compliance_filter:
                ok, c_reason = check_compliance(answer)
                if not ok:
                    # 不直接丢弃（丢弃会丢信息），而是降级进人工审核队列
                    filter_stats[c_reason] += 1
                    all_data.append({
                        'question': question,
                        'answer': answer,
                        'department': department,
                        'scene': scene,
                        'scene_name': scene_name,
                        'need_review': True,
                        'review_reason': c_reason,
                    })
                    continue
            
            # --- 新增层 C：不确定转人工 ---
            need_review, r_reason = needs_human_review(question, answer, scene_hits)
            if need_review:
                filter_stats["需人工复核:" + r_reason] += 1
            
            all_data.append({
                'question': question,
                'answer': answer,
                'department': department,
                'scene': scene,
                'scene_name': scene_name,
                'need_review': need_review,
                'review_reason': r_reason,
            })
            filter_stats["有效"] += 1
    
    print(f"\n--- 过滤统计 ---")
    for reason, count in filter_stats.most_common():
        print(f"  {reason}: {count} 条")
    
    # 去重
    all_data = deduplicate_by_question(all_data)
    
    train_ready, review_queue = partition_by_review(all_data)
    print(f"\n清洗完成: {len(all_data)} 条有效数据")
    print(f"  可直接训练: {len(train_ready)} 条")
    print(f"  待人工审核: {len(review_queue)} 条")
    return all_data


def partition_by_review(data_list):
    """
    按 need_review 标记拆分：可直接进训练集的 / 需人工审核的。
    返回 (train_ready, review_queue)
    """
    train_ready = [d for d in data_list if not d.get("need_review")]
    review_queue = [d for d in data_list if d.get("need_review")]
    return train_ready, review_queue


# ========================================
# 第三步：数据格式化（转为训练格式）
# ========================================

def _resolve_prompt(system_prompt):
    """显式传入 system_prompt 时优先，否则使用健康管理助手默认人设。"""
    return system_prompt or SYSTEM_PROMPT


def to_alpaca_format(data_list, system_prompt=None):
    """
    转换为 Alpaca 格式，保留 department 字段用于质量评估中的多样性分析，
    并透传 scene / need_review。
    system_prompt=None 时默认使用 health_schema.SYSTEM_PROMPT（健康管理助手）。
    """
    prompt = _resolve_prompt(system_prompt)
    alpaca_data = []
    for item in data_list:
        entry = {
            "instruction": prompt,
            "input": item['question'],
            "output": item['answer'],
        }
        if 'department' in item:
            entry["department"] = item['department']
        for key in ('scene', 'scene_name', 'need_review', 'review_reason'):
            if key in item:
                entry[key] = item[key]
        alpaca_data.append(entry)
    return alpaca_data


def to_chat_format(data_list, system_prompt=None):
    """
    转换为Chat对话格式（适用于Qwen等Chat模型），人设解析规则同 to_alpaca_format
    """
    prompt = _resolve_prompt(system_prompt)
    chat_data = []
    for item in data_list:
        chat_data.append({
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": item['question']},
                {"role": "assistant", "content": item['answer']},
            ]
        })
    return chat_data


def save_dataset(data, output_path, format_type="jsonl"):
    """保存数据集"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    if format_type == "jsonl":
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    elif format_type == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"数据集已保存: {output_path} ({len(data)} 条)")


# ========================================
# 第四步：数据采样（用于快速实验）
# ========================================

def sample_balanced_data(data_list, samples_per_dept=100, seed=42):
    """
    按科室均衡采样，用于快速实验
    """
    import random
    random.seed(seed)
    
    dept_data = defaultdict(list)
    for item in data_list:
        dept_data[item['department']].append(item)
    
    sampled = []
    for dept, items in dept_data.items():
        n = min(samples_per_dept, len(items))
        sampled.extend(random.sample(items, n))
        print(f"  {dept}: 采样 {n} 条")
    
    random.shuffle(sampled)
    print(f"均衡采样完成: {len(sampled)} 条")
    return sampled


# ========================================
# 主流程
# ========================================

def main():
    # 配置数据目录（根据实际路径修改）
    data_dir = "【数据集】中文医疗数据"
    output_dir = "processed_data"
    
    # Step 1: 数据收集
    print("=" * 60)
    print("Step 1: 数据收集")
    print("=" * 60)
    raw_data, collect_stats = collect_medical_data(data_dir)
    
    if not raw_data:
        print("[错误] 没有收集到任何数据")
        return
    
    # Step 2: 数据清洗
    print("\n" + "=" * 60)
    print("Step 2: 数据清洗")
    print("=" * 60)
    cleaned_data = clean_medical_data(raw_data)
    
    # Step 3: 格式转换与保存
    print("\n" + "=" * 60)
    print("Step 3: 格式转换与保存")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存Alpaca格式（完整）
    alpaca_data = to_alpaca_format(cleaned_data)
    save_dataset(alpaca_data, os.path.join(output_dir, "medical_alpaca_full.jsonl"))
    
    # 保存Chat格式（完整）
    chat_data = to_chat_format(cleaned_data)
    save_dataset(chat_data, os.path.join(output_dir, "medical_chat_full.jsonl"))
    
    # Step 3.5: 场景分布 + 人工审核队列导出（健康管理门诊新增）
    print("\n" + "=" * 60)
    print("Step 3.5: 场景分布与人工审核队列")
    print("=" * 60)
    scene_counter = Counter(item.get('scene_name', '未分类') for item in cleaned_data)
    for scene_name, count in scene_counter.most_common():
        print(f"  {scene_name}: {count} 条")
    train_ready, review_queue = partition_by_review(cleaned_data)
    save_dataset(review_queue, os.path.join(output_dir, "medical_review_queue.jsonl"))
    print(f"  待人工审核队列 -> {output_dir}/medical_review_queue.jsonl "
          f"（{len(review_queue)} 条，不进训练集）")

    # Step 4: 均衡采样（用于快速实验）
    print("\n" + "=" * 60)
    print("Step 4: 均衡采样（快速实验用）")
    print("=" * 60)
    # 采样只在"可直接训练"的样本里做，避免把需复核样本混进训练集
    sampled_data = sample_balanced_data(train_ready, samples_per_dept=200)
    sampled_alpaca = to_alpaca_format(sampled_data)
    save_dataset(sampled_alpaca, os.path.join(output_dir, "medical_alpaca_sampled.jsonl"))
    
    # 留出验证集（5%）
    val_size = max(10, len(sampled_data) // 20)
    val_data = to_alpaca_format(sampled_data[:val_size])
    train_data = to_alpaca_format(sampled_data[val_size:])
    save_dataset(val_data, os.path.join(output_dir, "medical_alpaca_val.jsonl"))
    save_dataset(train_data, os.path.join(output_dir, "medical_alpaca_train.jsonl"))
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"完整数据集: {len(alpaca_data)} 条")
    print(f"采样数据集: {len(sampled_alpaca)} 条")
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")


if __name__ == "__main__":
    main()
