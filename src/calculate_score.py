#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calculate_score.py
用法：python calculate_score.py <姓名>
      python calculate_score.py 小明

流程：
  1. 在 results/corrected/ 下所有 JSON 中的 students 字段查找该人
  2. 对每张证书，用编辑距离在 scoring_rules.json 中匹配最合适条目
  3. 依据 scoring_matrix 与排名计算加分区间
  4. 汇总所有奖项总分区间（含去重、类别上限）
  5. 将结果写入 results/students_score/<姓名>_<时间戳>.json
"""

import re
import json
import argparse
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import Optional

# ─── 路径配置 ────────────────────────────────────────────────────────────────

BASE_DIR           = Path("/Users/zhongzuodong/AutoCert")
CORRECTED_DIR      = BASE_DIR / "results" / "corrected"
SCORING_RULES_PATH = BASE_DIR / "roster" / "scoring_rules.json"
OUTPUT_DIR         = BASE_DIR / "results" / "students_score"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── 1. 基础工具 ─────────────────────────────────────────────────────────────

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
    s = re.sub(r'\d{4}(?:年|—|-)?', '', s)
    s = re.sub(r'（[^）]*）', '', s)
    s = re.sub(r'\([^\)]*\)', '', s)
    s = re.sub(r'\s+', '', s)
    return s.strip()


def normalize_award_level(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    mapping = {"一等奖": "一等", "二等奖": "二等", "三等奖": "三等",
               "一等":   "一等", "二等":   "二等", "三等":   "三等"}
    return mapping.get(raw.strip())


# ─── 2. 数据加载 ─────────────────────────────────────────────────────────────

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


# ─── 3. 规则扁平化 ───────────────────────────────────────────────────────────

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


# ─── 4. 智能匹配 ─────────────────────────────────────────────────────────────

TYPE_PRIORITY = {
    "Software Copyright": ["专利软著"],
    "Patent":             ["专利软著"],
}

def match_rule(cert_name: str, cert_type: str,
               flat_rules: list[dict]) -> tuple[Optional[dict], float]:
    cert_pre = preprocess_name(cert_name)

    priority_ids = TYPE_PRIORITY.get(cert_type, [])
    if priority_ids:
        candidates = [r for r in flat_rules
                      if r.get("competition_id") in priority_ids
                      or r.get("subcategory_id") in priority_ids]
        if cert_type == "Software Copyright":
            for r in candidates:
                if r["display_name"] == "软件著作权":
                    return r, 1.0
        if candidates:
            best = max(candidates,
                       key=lambda r: similarity_ratio(cert_pre,
                                                      preprocess_name(r["display_name"])))
            return best, similarity_ratio(cert_pre, preprocess_name(best["display_name"]))

    best_rule, best_sim = None, -1.0
    for rule in flat_rules:
        sim_display = similarity_ratio(cert_pre, preprocess_name(rule["display_name"]))
        sim_comp    = similarity_ratio(cert_pre, preprocess_name(rule.get("competition_name", "")))
        sim = max(sim_display, sim_comp)
        if sim > best_sim:
            best_sim, best_rule = sim, rule

    return best_rule, best_sim


# ─── 5. 加分计算 ─────────────────────────────────────────────────────────────

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
    out = dict(
        scoring_type="unknown",
        individual_max=None,
        team_total=None,
        score_range=None,
        score_fixed=None,
        award_rank_used=award_rank_norm,
        note="",
    )

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
                f"无对应数据（可能为校级三等、或奖项等级未能识别）"
            )
        return out

    ind_from_matrix = cell.get("individual")
    team_total      = cell.get("team_total")
    individual_max  = preset_ind if preset_ind is not None else ind_from_matrix
    out["individual_max"] = individual_max

    if rule.get("is_team") and team_total is not None:
        out["scoring_type"] = "matrix_team"
        out["team_total"]   = team_total
        raw_max = round(team_total / student_rank, 4)
        raw_min = round(team_total / total_students, 4)
        if individual_max is not None:
            raw_max = min(raw_max, individual_max)
            raw_min = min(raw_min, individual_max)
        out["score_range"] = [raw_min, raw_max]
    else:
        out["scoring_type"] = "matrix_individual"
        out["score_fixed"]   = individual_max

    return out


# ─── 6. 构造输出记录 ─────────────────────────────────────────────────────────

def build_entry(name: str, cert: dict,
                rule: dict, similarity: float,
                award_rank_norm: Optional[str],
                score: dict,
                student_rank: int, total_students: int) -> dict:

    is_team = rule.get("is_team", False)

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
        "matched_rule": {
            "display_name":     rule["display_name"],
            "competition_id":   rule.get("competition_id", ""),   # ← 新增，汇总去重需要
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


# ─── 7. 分数汇总 ─────────────────────────────────────────────────────────────

def aggregate_scores(entries: list[dict]) -> dict:
    """
    按加分规则汇总所有奖项，返回总分区间及过程说明。

    处理顺序（顺序不可颠倒）：
      ① 过滤无法计算的条目（scoring_type == "unknown"）
      ② 同年度同竞赛去重：key=(year, competition_id)，保留 score_max 最大的条目
      ③ 类别上限裁剪：艺术类 / 荣誉称号 各累计不超过 1.0 分
      ④ 线性加总 → [total_min, total_max]
    """

    # ── ① 提取可计算条目 ────────────────────────────────────────
    computable: list[dict] = []   # 可计算的中间结构
    skipped:    list[dict] = []   # 被跳过的条目

    for e in entries:
        s  = e["scoring"]
        st = s["scoring_type"]

        # 提取得分区间
        if st == "matrix_team":
            sc_min = s.get("score_range_min")
            sc_max = s.get("score_range_max")
        elif st in ("fixed", "matrix_individual"):
            val    = s.get("score_fixed") or s.get("individual_max")
            sc_min = sc_max = val
        else:
            # unknown
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
        score_cap      = e["matched_rule"].get("score_cap")       # None 表示无上限

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

    # ── ② 同年度同竞赛去重 ──────────────────────────────────────
    # key = (year, competition_id)；保留 score_max 最大的一条
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

        # 按 score_max 降序，score_min 次之
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
                    f"同年度({year})同竞赛类别({comp_id or '未知'})已有更高分项，"
                    f"按规则取最高分"
                ),
                "overridden_by": winner["certificate_name"],
            })

        dedup_groups.append({
            "year":             year,
            "competition_id":   comp_id,
            "kept":             winner["certificate_name"],
            "kept_score_max":   winner["score_max"],
            "dropped": [
                {"certificate_name": l["certificate_name"],
                 "score_max": l["score_max"]}
                for l in losers
            ],
        })

    # ── ③ 类别上限裁剪 ───────────────────────────────────────────
    # 按 category 分组，累计不超过 score_cap
    cap_notes:    list[dict] = []
    after_cap:    list[dict] = []

    # 先将有 cap 的 category 分组，其余直接通过
    by_category: dict = defaultdict(list)
    for item in after_dedup:
        by_category[item["category"]].append(item)

    for cat, items in by_category.items():
        cap = items[0]["score_cap"]     # 同 category 的 cap 值相同
        if cap is None:
            after_cap.extend(items)
            continue

        # 有 cap：按 score_max 从大到小贪心填充
        items_sorted = sorted(items, key=lambda x: x["score_max"], reverse=True)
        cumulative_max = 0.0
        cumulative_min = 0.0

        for item in items_sorted:
            if round(cumulative_max, 6) >= cap:
                # 已经到达上限，该条目完全跳过
                skipped.append({
                    "source_file":      item["source_file"],
                    "certificate_name": item["certificate_name"],
                    "reason": (
                        f"{cat}类加分已达上限({cap}分)，"
                        f"本条目不计入总分"
                    ),
                })
                continue

            remaining = round(cap - cumulative_max, 6)
            if item["score_max"] > remaining:
                # 需要裁剪：按比例同步缩小 score_min
                ratio    = remaining / item["score_max"] if item["score_max"] else 0
                capped_max = round(remaining, 4)
                capped_min = round(item["score_min"] * ratio, 4)

                cap_notes.append({
                    "certificate_name": item["certificate_name"],
                    "category":         cat,
                    "cap":              cap,
                    "original_min":     item["score_min"],
                    "original_max":     item["score_max"],
                    "capped_min":       capped_min,
                    "capped_max":       capped_max,
                    "reason":           f"{cat}类累计上限{cap}分，本条目按剩余空间裁剪",
                })

                item = dict(item)       # 浅拷贝，不修改原始 entry
                item["score_min"] = capped_min
                item["score_max"] = capped_max

            cumulative_max = round(cumulative_max + item["score_max"], 6)
            cumulative_min = round(cumulative_min + item["score_min"], 6)
            after_cap.append(item)

    # ── ④ 线性加总 ──────────────────────────────────────────────
    total_min = round(sum(item["score_min"] for item in after_cap), 4)
    total_max = round(sum(item["score_max"] for item in after_cap), 4)

    # ── 构造返回结构 ─────────────────────────────────────────────
    return {
        # 核心结论
        "total_score_min": total_min,
        "total_score_max": total_max,
        "total_score_summary": (
            f"总加分区间：[{total_min}, {total_max}]"
            if total_min != total_max
            else f"总加分：{total_max}"
        ),

        # 统计
        "effective_count": len(after_cap),
        "skipped_count":   len(skipped),

        # 实际参与计分的条目（含可能被裁剪后的分值）
        "effective_entries": [
            {
                "certificate_name": item["certificate_name"],
                "source_file":      item["source_file"],
                "year":             item["year"],
                "category":         item["category"],
                "competition_id":   item["competition_id"],
                "score_min":        item["score_min"],
                "score_max":        item["score_max"],
            }
            for item in after_cap
        ],

        # 被跳过的条目及原因
        "skipped_entries": skipped,

        # 去重说明
        "dedup_notes": dedup_groups,

        # 上限裁剪说明
        "cap_notes": cap_notes,

        # 提示
        "warnings": [
            "⚠ 总分为估算区间，最终以学院学术委员会审核为准。",
            "⚠ 同一科技作品获两个不同奖项仅取最高分，需人工核查（系统未自动识别）。",
            "⚠ 论文/专利/软著需核查署名单位为「北京科技大学」。",
            "⚠ 匹配置信度为「低」的条目请人工复核后再计分。",
        ],
    }


# ─── 8. 主流程 ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="保研加分计算器")
    parser.add_argument("name", help="要查询的学生姓名，例如：马金瑶")
    args   = parser.parse_args()
    name: str = args.name.strip()

    rules          = load_json(SCORING_RULES_PATH)
    scoring_matrix = rules.get("scoring_matrix", {})
    flat_rules     = flatten_rules(rules)
    records        = load_corrected_records(CORRECTED_DIR)

    print(f"\n{'─'*55}")
    print(f"  🔍 查询人：{name}")
    print(f"{'─'*55}")

    matched_certs = [
        r for r in records
        if name in r.get("students", [])
    ]

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

        best_rule, sim = match_rule(cert_name, cert_type, flat_rules)
        if best_rule is None:
            print(f"  [WARN] 无法匹配：{cert_name}")
            continue

        confidence = "高" if sim >= 0.7 else "中" if sim >= 0.4 else "低⚠"
        print(f"  📄 {cert_name}")
        print(f"     → 匹配：{best_rule['display_name']}"
              f"  (相似度={sim:.3f}, 置信={confidence})")

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
                        award_norm, score, rank, total)
        )

    # ── 汇总总分 ──────────────────────────────────────────────────
    summary = aggregate_scores(entries)

    print(f"{'─'*55}")
    print(f"  📊 {summary['total_score_summary']}")
    print(f"     参与计分：{summary['effective_count']} 项"
          f"  |  跳过：{summary['skipped_count']} 项")
    if summary["dedup_notes"]:
        print(f"     去重处理：{len(summary['dedup_notes'])} 组")
    if summary["cap_notes"]:
        print(f"     上限裁剪：{len(summary['cap_notes'])} 条")
    print(f"{'─'*55}")

    # ── 输出 JSON ────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = OUTPUT_DIR / f"{name}_{timestamp}.json"
    output_doc = {
        "student_name":             name,
        "generated_at":             datetime.now().isoformat(timespec="seconds"),
        "total_certificates_found": len(entries),
        # ── 汇总结论放在最前面，便于快速查阅 ──
        "score_summary":            summary,
        # ── 逐条明细 ──
        "score_details":            entries,
        "global_rules_reminder":    rules.get("global_notes", []),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_doc, f, ensure_ascii=False, indent=2)

    print(f"  ✅ 结果已写入：{out_path}")
    print(f"{'─'*55}")
    print("  ⚠  提示：同年度同类别请取最高分，软著/论文需核查署名单位。")
    print(f"{'─'*55}\n")


if __name__ == "__main__":
    main()
