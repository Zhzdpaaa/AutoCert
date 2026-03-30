#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calculate_score.py (修改版)
新增：级别关键词识别模块，辅助编辑距离进行精准匹配
"""

import re
import json
import argparse
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# ─── 路径配置 ────────────────────────────────────────────────────────────────

BASE_DIR           = Path("/mnt/tidal-alsh01/dataset/OCRData/page_label_project/AutoCert")
CORRECTED_DIR      = BASE_DIR / "results" / "corrected" / "tianxuanzhizi"
SCORING_RULES_PATH = BASE_DIR / "roster" / "scoring_rules.json"
OUTPUT_DIR         = BASE_DIR / "results" / "students_score"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 【新增模块】级别关键词识别
# ══════════════════════════════════════════════════════════════════════════════

# ─── 省份/地区名称 ───────────────────────────────────────────────────────────

PROVINCES = [
    "北京", "上海", "天津", "重庆",
    "河北", "山西", "辽宁", "吉林", "黑龙江",
    "江苏", "浙江", "安徽", "福建", "江西",
    "山东", "河南", "湖北", "湖南", "广东",
    "海南", "四川", "贵州", "云南", "陕西",
    "甘肃", "青海", "台湾", "内蒙古", "广西",
    "西藏", "宁夏", "新疆", "香港", "澳门"
]

REGIONS = ["华北", "华东", "华南", "华中", "西北", "西南", "东北", "长三角", "珠三角"]

# ─── 级别关键词配置 ─────────────────────────────────────────────────────────

# 国家级关键词（按优先级排序，长词优先匹配）
NATIONAL_KEYWORDS = [
    # 最高优先级：明确的全国决赛
    "全国总决赛", "全国决赛", "国赛总决赛", "国赛决赛",
    # 高优先级：全国性赛事
    "全国赛", "国赛", "国家级",
    # 中等优先级
    "中国区决赛", "中国赛区决赛",
]

# 省/市级关键词
PROVINCIAL_KEYWORDS = [
    # 明确的省/市赛
    "省赛", "市赛", "省级", "市级",
    # 赛区相关
    "赛区", "分赛区", "区域赛", "分区赛", "大区赛",
    # 选拔赛（通常是省市级）
    "省选", "市选", "地区选拔",
]

# 校级关键词
SCHOOL_KEYWORDS = [
    "校赛", "校级", "校内选拔赛", "校内选拔", "校内赛", "校内",
    "院赛", "院级", "院内",
    "选拔赛",  # 注意：单独的"选拔赛"可能是校内选拔，优先级较低
]


def extract_competition_level(cert_name: str) -> Tuple[str, str]:
    """
    从证书名称中提取比赛级别。
    
    Args:
        cert_name: 证书名称
        
    Returns:
        (级别, 匹配到的关键词)
        级别为: "国家级", "省/市级", "校级", "未知"
    """
    name = cert_name.strip()
    
    # ─── 步骤1：检测明确的全国决赛（最高优先级） ───────────────────────
    # 这些关键词明确表示是国赛，即使同时包含赛区信息也应判定为国赛
    definite_national = ["全国总决赛", "全国决赛", "国赛总决赛", "国赛决赛", "总决赛"]
    for kw in definite_national:
        if kw in name:
            # 检查"总决赛"后面是否紧跟赛区（如"总决赛北京赛区"这种矛盾写法）
            idx = name.find(kw)
            after = name[idx + len(kw):]
            # 如果后面没有省/市赛区标识，则判定为国家级
            has_regional_after = any(p + "赛区" in after or p + "分赛区" in after 
                                     for p in PROVINCES + REGIONS)
            if not has_regional_after:
                return "国家级", kw
    
    # ─── 步骤2：检测省/市级标识（第二优先级） ─────────────────────────
    # 2.1 检查 "XX赛区" 模式
    for province in PROVINCES:
        patterns = [
            f"{province}赛区", f"{province}省赛区", f"{province}市赛区",
            f"{province}省赛", f"{province}市赛",
            f"（{province}）", f"({province})",  # 括号内的省份名
        ]
        for pat in patterns:
            if pat in name:
                return "省/市级", pat
    
    for region in REGIONS:
        patterns = [f"{region}赛区", f"{region}大区", f"{region}分赛区"]
        for pat in patterns:
            if pat in name:
                return "省/市级", pat
    
    # 2.2 检查通用省/市级关键词
    for kw in PROVINCIAL_KEYWORDS:
        if kw in name:
            return "省/市级", kw
    
    # ─── 步骤3：检测校级标识（第三优先级） ─────────────────────────────
    for kw in SCHOOL_KEYWORDS:
        if kw in name:
            # "选拔赛"需要额外检查是否是"省选拔赛"之类
            if kw == "选拔赛":
                # 检查前面是否有省份名
                idx = name.find(kw)
                before = name[:idx]
                if any(p in before[-5:] for p in PROVINCES):  # 检查关键词前5个字符
                    return "省/市级", f"{before[-2:]}{kw}"
            return "校级", kw
    
    # ─── 步骤4：检测国家级标识（最低优先级） ─────────────────────────────
    for kw in NATIONAL_KEYWORDS:
        if kw in name:
            # 二次检查：确保不包含明显的省市级标识
            has_provincial = any(pk in name for pk in PROVINCIAL_KEYWORDS)
            has_province_name = any(p + "赛区" in name for p in PROVINCES)
            if not has_provincial and not has_province_name:
                return "国家级", kw
    
    return "未知", ""


def normalize_rule_level(rule_level: Optional[str]) -> str:
    """
    将规则的 level 字段归一化为统一级别类别。
    
    示例输入 → 输出：
    - "国家级A", "国家级B" → "国家级"
    - "市级A", "市级B", "省级A" → "省/市级"  
    - "校级A", "校级B" → "校级"
    - "固定", None → "固定", "未知"
    """
    if not rule_level:
        return "未知"
    
    level = rule_level.strip()
    
    if "国家" in level or "国" in level and "级" in level:
        return "国家级"
    elif "省" in level or "市" in level:
        return "省/市级"
    elif "校" in level:
        return "校级"
    elif level == "固定":
        return "固定"
    else:
        return "未知"


def get_level_match_modifier(cert_level: str, rule_level: str) -> Tuple[float, str]:
    """
    计算级别匹配的分数修正值。
    
    Returns:
        (修正值, 说明)
        - 级别完全匹配：+0.20 加成
        - 级别相邻（如省级证书匹配校级规则）：-0.15 轻度惩罚
        - 级别相差2级（如国家级证书匹配校级规则）：-0.35 重度惩罚
    """
    if cert_level == "未知" or rule_level in ("未知", "固定"):
        return 0.0, "级别未知，无修正"
    
    level_order = {"国家级": 3, "省/市级": 2, "校级": 1}
    
    cert_rank = level_order.get(cert_level, 0)
    rule_rank = level_order.get(rule_level, 0)
    
    diff = abs(cert_rank - rule_rank)
    
    if diff == 0:
        return 0.20, "级别匹配"
    elif diff == 1:
        return -0.15, "级别相邻"
    else:  # diff >= 2
        return -0.35, "级别相差过大"


# ══════════════════════════════════════════════════════════════════════════════
# 原有工具函数（保持不变）
# ══════════════════════════════════════════════════════════════════════════════

def levenshtein(s1: str, s2: str) -> int:
    if s1 == s2:
        return 0
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def similarity_ratio(s1: str, s2: str) -> float:
    denom = max(len(s1), len(s2), 1)
    return 1.0 - levenshtein(s1, s2) / denom


def preprocess_name(s: str) -> str:
    """预处理名称：移除年份、括号内容、空白"""
    s = re.sub(r'\d{4}(?:年|—|-)?', '', s)
    s = re.sub(r'（[^）]*）', '', s)
    s = re.sub(r'\([^\)]*\)', '', s)
    s = re.sub(r'\s+', '', s)
    return s.strip()


def preprocess_name_keep_level(s: str) -> str:
    """预处理名称：移除年份、空白，但保留括号内容（可能包含级别信息）"""
    s = re.sub(r'\d{4}(?:年|—|-)?', '', s)
    s = re.sub(r'\s+', '', s)
    return s.strip()


def normalize_award_level(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    mapping = {"一等奖": "一等", "二等奖": "二等", "三等奖": "三等",
               "一等":   "一等", "二等":   "二等", "三等":   "三等",
               "金奖": "一等", "银奖": "二等", "铜奖": "三等",
               "特等奖": "特等", "优秀奖": "优秀"}
    return mapping.get(raw.strip())


# ══════════════════════════════════════════════════════════════════════════════
# 数据加载 & 规则扁平化（保持不变）
# ══════════════════════════════════════════════════════════════════════════════

def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_corrected_records(directory: Path) -> list[dict]:
    records = []
    for fp in sorted(directory.glob("*.json")):
        try:
            data = load_json(fp)
            data["_source_file"] = fp.name
            records.append(data)
        except Exception as exc:
            print(f"[WARN] 跳过 {fp.name}：{exc}")
    return records


def flatten_rules(rules: dict) -> list[dict]:
    flat = []

    def _add(entry_name, comp, entry_override, cat, sub):
        flat.append({
            "display_name":     entry_override.get("name", entry_name),
            "level":            entry_override.get("level",   comp.get("level")),
            "award_rank":       entry_override.get("award_rank", comp.get("award_rank")),
            "is_team":          entry_override.get("is_team",  comp.get("is_team", False)),
            "fixed_score":      entry_override.get("fixed_score", comp.get("fixed_score")),
            "individual_score": entry_override.get("individual_score"),
            "category_id":      cat.get("id", ""),
            "category_name":    cat.get("name", ""),
            "subcategory_id":   sub.get("id", ""),
            "subcategory_name": sub.get("name", ""),
            "competition_id":   comp.get("id", comp.get("name", "")),
            "competition_name": comp.get("name", ""),
            "special_rule":     comp.get("special_rule", sub.get("special_rule", "")),
            "score_cap":        cat.get("score_cap"),
        })

    for cat in rules.get("categories", []):
        for sub in cat.get("subcategories", []):
            for comp in sub.get("competitions", []):
                entries = comp.get("entries")
                if entries:
                    for e in entries:
                        _add(comp.get("name", ""), comp, e, cat, sub)
                else:
                    _add(comp.get("name", ""), comp, {}, cat, sub)

    return flat


# ══════════════════════════════════════════════════════════════════════════════
# 【重点修改】智能匹配函数
# ══════════════════════════════════════════════════════════════════════════════

TYPE_PRIORITY = {
    "Software Copyright": ["专利软著"],
    "Patent":             ["专利软著"],
}


def match_rule(cert_name: str, cert_type: str,
               flat_rules: list[dict]) -> Tuple[Optional[dict], float, dict]:
    """
    智能匹配规则，结合级别关键词识别和编辑距离。
    
    Args:
        cert_name: 证书名称
        cert_type: 证书类型
        flat_rules: 扁平化的规则列表
        
    Returns:
        (最佳匹配规则, 文本相似度, 匹配调试信息)
    """
    cert_pre = preprocess_name(cert_name)
    
    # 提取证书级别
    cert_level, level_keyword = extract_competition_level(cert_name)
    
    debug_info = {
        "cert_level_detected": cert_level,
        "level_keyword_found": level_keyword,
        "candidates_considered": [],
    }
    
    # ─── 特殊类型处理（软著/专利） ─────────────────────────────────────
    priority_ids = TYPE_PRIORITY.get(cert_type, [])
    if priority_ids:
        candidates = [r for r in flat_rules
                      if r.get("competition_id") in priority_ids
                      or r.get("subcategory_id") in priority_ids]
        if cert_type == "Software Copyright":
            for r in candidates:
                if r["display_name"] == "软件著作权":
                    return r, 1.0, debug_info
        if candidates:
            best = max(candidates,
                       key=lambda r: similarity_ratio(cert_pre,
                                                      preprocess_name(r["display_name"])))
            sim = similarity_ratio(cert_pre, preprocess_name(best["display_name"]))
            return best, sim, debug_info
    
    # ─── 计算综合匹配分数 ─────────────────────────────────────────────
    scored_rules = []
    
    for rule in flat_rules:
        # 文本相似度（使用两种预处理方式，取较高值）
        sim_display_1 = similarity_ratio(cert_pre, preprocess_name(rule["display_name"]))
        sim_display_2 = similarity_ratio(
            preprocess_name_keep_level(cert_name),
            preprocess_name_keep_level(rule["display_name"])
        )
        sim_comp = similarity_ratio(cert_pre, preprocess_name(rule.get("competition_name", "")))
        text_sim = max(sim_display_1, sim_display_2, sim_comp)
        
        # 级别匹配修正
        rule_level = normalize_rule_level(rule.get("level"))
        level_mod, level_note = get_level_match_modifier(cert_level, rule_level)
        
        # 综合分数 = 文本相似度 + 级别修正
        final_score = text_sim + level_mod
        
        scored_rules.append({
            "rule": rule,
            "text_sim": text_sim,
            "rule_level": rule_level,
            "level_modifier": level_mod,
            "level_note": level_note,
            "final_score": final_score,
        })
    
    # 按综合分数排序
    scored_rules.sort(key=lambda x: x["final_score"], reverse=True)
    
    # 记录前5个候选（用于调试）
    debug_info["candidates_considered"] = [
        {
            "display_name": sr["rule"]["display_name"],
            "rule_level": sr["rule_level"],
            "text_sim": round(sr["text_sim"], 4),
            "level_mod": sr["level_modifier"],
            "final_score": round(sr["final_score"], 4),
        }
        for sr in scored_rules[:5]
    ]
    
    if scored_rules:
        best = scored_rules[0]
        return best["rule"], best["text_sim"], debug_info
    
    return None, 0.0, debug_info

# ══════════════════════════════════════════════════════════════════════════════
# 【修改】团队分数区间计算函数
# ══════════════════════════════════════════════════════════════════════════════

def compute_team_score_range(
    team_total: float,
    total_students: int,
    student_rank: int,
    individual_max: Optional[float] = None
) -> Tuple[float, float]:
    """
    【新增函数】计算团队中某个排名位置的分数区间
    
    数学推导：
    - 约束：s₁ ≥ s₂ ≥ ... ≥ sₙ ≥ 0 且 Σsᵢ = T 且 sᵢ ≤ M
    - 最大值策略：让前 k 人平分，后面的人拿 0
    - 最小值策略：让前 k-1 人各拿满 M，剩余由后 n-k+1 人平分
    
    Args:
        team_total: 团队总分 T
        total_students: 团队人数 n
        student_rank: 当前位置 k（从 1 开始）
        individual_max: 个人上限 M（None 表示无上限）
    
    Returns:
        (score_min, score_max) 分数区间
    """
    T = team_total
    n = total_students
    k = student_rank
    M = individual_max if individual_max is not None else T  # 无上限时设为总分
    
    # ─── 最大值计算 ───────────────────────────────────────────
    # 策略：让前 k 人平分所有分数，后面的人拿 0
    # max(sₖ) = min(T/k, M)
    score_max = min(T / k, M)
    
    # ─── 最小值计算 ───────────────────────────────────────────
    # 策略：让前 k-1 人各拿 M，剩余由后 n-k+1 人平分
    # min(sₖ) = max(0, (T - (k-1)*M) / (n-k+1))
    remaining = T - (k - 1) * M          # 前 k-1 人拿满后的剩余
    share_count = n - k + 1              # 后面包括自己在内的人数
    score_min = 0.0
    
    return round(score_min, 4), round(score_max, 4)

# ══════════════════════════════════════════════════════════════════════════════
# 加分计算（保持原有逻辑）
# ══════════════════════════════════════════════════════════════════════════════

def get_matrix_cell(scoring_matrix: dict,
                    level: str,
                    award_rank: Optional[str]) -> Optional[dict]:
    if not level or not award_rank:
        return None
    level_data = scoring_matrix.get(level)
    if level_data is None:
        return None
    return level_data.get(award_rank)


def compute_score(rule: dict,
                  award_rank_norm: Optional[str],
                  scoring_matrix: dict,
                  student_rank: int,
                  total_students: int) -> dict:
    """
    【修改函数】计算加分，使用精确的区间公式
    """
    out = dict(
        scoring_type="unknown",
        individual_max=None,
        team_total=None,
        score_range=None,
        score_fixed=None,
        award_rank_used=award_rank_norm,
        note="",
    )

    # ─── 固定分数处理 ─────────────────────────────────────────────────
    if rule.get("fixed_score") is not None and rule.get("level") in ("固定", None):
        out["scoring_type"] = "fixed"
        out["score_fixed"]   = rule["fixed_score"]
        out["individual_max"]= rule["fixed_score"]
        return out

    rule_ar = rule.get("award_rank")
    effective_rank = award_rank_norm or rule_ar
    out["award_rank_used"] = effective_rank

    preset_ind = rule.get("individual_score")
    level = rule.get("level", "")
    cell  = get_matrix_cell(scoring_matrix, level, effective_rank)

    if cell is None:
        if rule.get("fixed_score") is not None:
            out["scoring_type"] = "fixed"
            out["score_fixed"]   = rule["fixed_score"]
            out["individual_max"]= rule["fixed_score"]
        else:
            out["note"] = (
                f"scoring_matrix 中 level='{level}', rank='{effective_rank}' "
                f"无对应数据（可能奖项等级未能识别、或为校级三等）"
            )
        return out

    ind_from_matrix = cell.get("individual")
    team_total      = cell.get("team_total")
    individual_max  = preset_ind if preset_ind is not None else ind_from_matrix
    out["individual_max"] = individual_max

    # ─── 团队项目：使用精确公式计算区间 ───────────────────────────────
    if rule.get("is_team") and team_total is not None:
        out["scoring_type"] = "matrix_team"
        out["team_total"]   = team_total
        
        # ═══════════════════════════════════════════════════════════════
        # 【核心修改】使用精确的数学公式计算分数区间
        # ═══════════════════════════════════════════════════════════════
        score_min, score_max = compute_team_score_range(
            team_total=team_total,
            total_students=total_students,
            student_rank=student_rank,
            individual_max=individual_max
        )
        
        out["score_range"] = [score_min, score_max]
        
    else:
        out["scoring_type"] = "matrix_individual"
        out["score_fixed"]   = individual_max

    return out


# ══════════════════════════════════════════════════════════════════════════════
# 构造输出记录（增加级别识别信息）
# ══════════════════════════════════════════════════════════════════════════════

def build_entry(name: str, cert: dict,
                rule: dict, similarity: float,
                award_rank_norm: Optional[str],
                score: dict,
                student_rank: int, total_students: int,
                match_debug: dict) -> dict:

    is_team = rule.get("is_team", False)
    cert_level = match_debug.get("cert_level_detected", "未知")
    level_keyword = match_debug.get("level_keyword_found", "")

    def _summary():
        parts = ["[团队]" if is_team else "[个人]"]
        parts.append(rule["display_name"])
        parts.append(f"({rule.get('level','')})")
        if score["award_rank_used"]:
            parts.append(f"· {score['award_rank_used']}奖")
        if is_team:
            parts.append(f"· 团队总分 {score['team_total']}")
            if score["score_range"]:
                lo, hi = score["score_range"]
                parts.append(
                    f"· {name} 排第 {student_rank}/{total_students} 位"
                    f"，可加分区间 [{lo}, {hi}]"
                )
        else:
            sc = score["score_fixed"] or score["individual_max"]
            if sc is not None:
                parts.append(f"· 可加 {sc} 分")
        if score["note"]:
            parts.append(f"⚠ {score['note']}")
        return " ".join(parts)

    return {
        "source_file":            cert.get("_source_file", ""),
        "certificate_type":       cert.get("certificate_type", ""),
        "certificate_name":       cert.get("name", ""),
        "issue_date":             cert.get("issue_date", ""),
        "issuing_authority":      cert.get("issuing_authority", []),
        "students_in_cert":       cert.get("students", []),
        "advisors_in_cert":       cert.get("advisors", []),
        "student_name":           name,
        "student_rank_in_team":   student_rank if is_team else None,
        "total_students_in_cert": total_students if is_team else None,
        # 新增：级别识别信息
        "level_detection": {
            "detected_level": cert_level,
            "keyword_found": level_keyword,
            "candidates_top3": match_debug.get("candidates_considered", [])[:3],
        },
        "matched_rule": {
            "display_name":     rule["display_name"],
            "competition_id":   rule.get("competition_id", ""),
            "competition_name": rule.get("competition_name", ""),
            "category":         rule.get("category_name", ""),
            "subcategory":      rule.get("subcategory_name", ""),
            "level":            rule.get("level", ""),
            "is_team":          is_team,
            "special_rule":     rule.get("special_rule", "") or None,
            "score_cap":        rule.get("score_cap"),
            "match_similarity": round(similarity, 4),
            "match_confidence": (
                "高" if similarity >= 0.7 else
                "中" if similarity >= 0.4 else "低（请人工复核）"
            ),
        },
        "award_level_raw":        cert.get("award_level"),
        "award_level_normalized": score["award_rank_used"],
        "scoring": {
            "scoring_type":    score["scoring_type"],
            "individual_max":  score["individual_max"],
            "team_total":      score["team_total"],
            "score_range_min": score["score_range"][0] if score["score_range"] else None,
            "score_range_max": score["score_range"][1] if score["score_range"] else None,
            "score_fixed":     score["score_fixed"],
            "note":            score["note"] or None,
        },
        "summary": _summary(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 分数汇总（保持原有逻辑）
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_scores(entries: list[dict]) -> dict:
    """按加分规则汇总所有奖项"""
    
    computable: list[dict] = []
    skipped:    list[dict] = []

    for e in entries:
        s  = e["scoring"]
        st = s["scoring_type"]

        if st == "matrix_team":
            sc_min = s.get("score_range_min")
            sc_max = s.get("score_range_max")
        elif st in ("fixed", "matrix_individual"):
            val    = s.get("score_fixed") or s.get("individual_max")
            sc_min = sc_max = val
        else:
            skipped.append({
                "source_file":      e["source_file"],
                "certificate_name": e["certificate_name"],
                "reason":           "scoring_type=unknown，奖项等级缺失或无法匹配",
                "note":             s.get("note"),
            })
            continue

        if sc_min is None or sc_max is None:
            skipped.append({
                "source_file":      e["source_file"],
                "certificate_name": e["certificate_name"],
                "reason":           "分数字段为 null，无法参与汇总",
                "note":             s.get("note"),
            })
            continue

        year           = (e.get("issue_date") or "")[:4] or "未知年份"
        competition_id = e["matched_rule"].get("competition_id", "")
        category       = e["matched_rule"].get("category", "")
        score_cap      = e["matched_rule"].get("score_cap")

        computable.append({
            "source_file":      e["source_file"],
            "certificate_name": e["certificate_name"],
            "year":             year,
            "competition_id":   competition_id,
            "category":         category,
            "score_cap":        score_cap,
            "score_min":        sc_min,
            "score_max":        sc_max,
        })

    # 同年度同竞赛去重
    dedup_groups: list[dict] = []
    groups: dict = defaultdict(list)

    for item in computable:
        key = (item["year"], item["competition_id"])
        groups[key].append(item)

    after_dedup: list[dict] = []

    for (year, comp_id), items in groups.items():
        if len(items) == 1:
            after_dedup.append(items[0])
            continue

        items_sorted = sorted(items,
                              key=lambda x: (x["score_max"], x["score_min"]),
                              reverse=True)
        winner = items_sorted[0]
        losers = items_sorted[1:]

        after_dedup.append(winner)

        for loser in losers:
            skipped.append({
                "source_file":      loser["source_file"],
                "certificate_name": loser["certificate_name"],
                "reason": (
                    f"同年度({year})同竞赛类别({comp_id or '未知'})已有更高分项"
                ),
                "overridden_by": winner["certificate_name"],
            })

        dedup_groups.append({
            "year":             year,
            "competition_id":   comp_id,
            "kept":             winner["certificate_name"],
            "kept_score_max":   winner["score_max"],
            "dropped": [{"certificate_name": l["certificate_name"], "score_max": l["score_max"]}
                        for l in losers],
        })

    # 类别上限裁剪
    cap_notes:    list[dict] = []
    after_cap:    list[dict] = []
    by_category: dict = defaultdict(list)
    
    for item in after_dedup:
        by_category[item["category"]].append(item)

    for cat, items in by_category.items():
        cap = items[0]["score_cap"]
        if cap is None:
            after_cap.extend(items)
            continue

        items_sorted = sorted(items, key=lambda x: x["score_max"], reverse=True)
        cumulative_max = 0.0

        for item in items_sorted:
            if round(cumulative_max, 6) >= cap:
                skipped.append({
                    "source_file":      item["source_file"],
                    "certificate_name": item["certificate_name"],
                    "reason": f"{cat}类加分已达上限({cap}分)",
                })
                continue

            remaining = round(cap - cumulative_max, 6)
            if item["score_max"] > remaining:
                ratio = remaining / item["score_max"] if item["score_max"] else 0
                capped_max = round(remaining, 4)
                capped_min = round(item["score_min"] * ratio, 4)

                cap_notes.append({
                    "certificate_name": item["certificate_name"],
                    "category": cat,
                    "original_max": item["score_max"],
                    "capped_max": capped_max,
                })

                item = dict(item)
                item["score_min"] = capped_min
                item["score_max"] = capped_max

            cumulative_max = round(cumulative_max + item["score_max"], 6)
            after_cap.append(item)

    # 线性加总
    total_min = round(sum(item["score_min"] for item in after_cap), 4)
    total_max = round(sum(item["score_max"] for item in after_cap), 4)

    return {
        "total_score_min": total_min,
        "total_score_max": total_max,
        "total_score_summary": (
            f"总加分区间：[{total_min}, {total_max}]"
            if total_min != total_max
            else f"总加分：{total_max}"
        ),
        "effective_count": len(after_cap),
        "skipped_count":   len(skipped),
        "effective_entries": [
            {
                "certificate_name": item["certificate_name"],
                "source_file":      item["source_file"],
                "year":             item["year"],
                "category":         item["category"],
                "score_min":        item["score_min"],
                "score_max":        item["score_max"],
            }
            for item in after_cap
        ],
        "skipped_entries": skipped,
        "dedup_notes": dedup_groups,
        "cap_notes": cap_notes,
        "warnings": [
            "⚠ 总分为估算区间，最终以学院学术委员会审核为准。",
            "⚠ 同一科技作品获两个不同奖项仅取最高分，需人工核查。",
            "⚠ 论文/专利/软著需核查署名单位为「北京科技大学」。",
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="保研加分计算器")
    parser.add_argument("name", help="要查询的学生姓名")
    args   = parser.parse_args()
    name: str = args.name.strip()

    rules          = load_json(SCORING_RULES_PATH)
    scoring_matrix = rules.get("scoring_matrix", {})
    flat_rules     = flatten_rules(rules)
    records        = load_corrected_records(CORRECTED_DIR)

    print(f"\n{'─'*60}")
    print(f"  🔍 查询人：{name}")
    print(f"{'─'*60}")

    matched_certs = [r for r in records if name in r.get("students", [])]

    if not matched_certs:
        print(f"[INFO] 未在任何证书的 students 字段中找到 '{name}'，退出。")
        return

    print(f"[INFO] 共找到 {len(matched_certs)} 张证书记录。\n")

    entries = []

    for cert in matched_certs:
        cert_name  = cert.get("name", "")
        cert_type  = cert.get("certificate_type", "")
        students   = cert.get("students", [])
        award_raw  = cert.get("award_level")
        award_norm = normalize_award_level(award_raw)

        try:
            rank = students.index(name) + 1
        except ValueError:
            rank = len(students)
        total = len(students)

        best_rule, sim, match_debug = match_rule(cert_name, cert_type, flat_rules)
        
        if best_rule is None:
            print(f"  [WARN] 无法匹配：{cert_name}")
            continue

        # 显示级别识别信息
        cert_level = match_debug.get("cert_level_detected", "未知")
        level_kw = match_debug.get("level_keyword_found", "")
        
        confidence = "高" if sim >= 0.7 else "中" if sim >= 0.4 else "低⚠"
        print(f"  📄 {cert_name}")
        print(f"     🏷️  级别识别：{cert_level}" + (f" (关键词: '{level_kw}')" if level_kw else ""))
        print(f"     → 匹配：{best_rule['display_name']} ({best_rule.get('level', '')})")
        print(f"        相似度={sim:.3f}, 置信={confidence}")

        score = compute_score(best_rule, award_norm, scoring_matrix, rank, total)

        if score["scoring_type"] == "matrix_team" and score["score_range"]:
            lo, hi = score["score_range"]
            print(f"     → 团队总分 {score['team_total']}，"
                  f"本人排名 {rank}/{total}，"
                  f"可加分区间 [{lo}, {hi}]")
        elif score["score_fixed"] is not None:
            print(f"     → 固定加分：{score['score_fixed']}")
        if score["note"]:
            print(f"     ⚠ {score['note']}")
        print()

        entries.append(
            build_entry(name, cert, best_rule, sim,
                        award_norm, score, rank, total, match_debug)
        )

    # 汇总总分
    summary = aggregate_scores(entries)

    print(f"{'─'*60}")
    print(f"  📊 {summary['total_score_summary']}")
    print(f"     参与计分：{summary['effective_count']} 项"
          f"  |  跳过：{summary['skipped_count']} 项")
    print(f"{'─'*60}")

    # 输出 JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = OUTPUT_DIR / f"{name}_{timestamp}.json"
    output_doc = {
        "student_name":             name,
        "generated_at":             datetime.now().isoformat(timespec="seconds"),
        "total_certificates_found": len(entries),
        "score_summary":            summary,
        "score_details":            entries,
        "global_rules_reminder":    rules.get("global_notes", []),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_doc, f, ensure_ascii=False, indent=2)

    print(f"  ✅ 结果已写入：{out_path}\n")


if __name__ == "__main__":
    main()
