#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3: 姓名校正模块
====================================================
功能:
  1. 用编辑距离将提取出的 students / advisors 与
     people_name.json 中的真实姓名做匹配校正
  2. 英文姓名自动映射为对应的中文姓名
  3. students 中实为 advisor 的条目自动移至 advisors
  4. 无法匹配的条目从列表中移除并记录说明
  5. 存在多个等距候选时写入 warnings，提醒人工确认
====================================================
"""

import json
import os
import sys
import argparse
import logging
from pathlib import Path
from copy import deepcopy
from typing import Optional


# ============================================================
# 1. 编辑距离
# ============================================================

def levenshtein(s1: str, s2: str) -> int:
    """计算两字符串的 Levenshtein 编辑距离（大小写不敏感）"""
    s1, s2 = s1.lower(), s2.lower()
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for c1 in s1:
        curr = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(
                prev[j + 1] + 1,          # 删除
                curr[j] + 1,               # 插入
                prev[j] + (c1 != c2)       # 替换
            ))
        prev = curr
    return prev[-1]


# ============================================================
# 2. 辅助工具
# ============================================================

def is_chinese(text: str) -> bool:
    """判断字符串中是否含有汉字"""
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)


def english_name_variants(name: str) -> list[str]:
    """
    生成英文姓名的顺序变体，处理「名 姓」与「姓 名」两种写法。
    例: "Jinyao Ma" → ["Jinyao Ma", "Ma Jinyao"]
         "Ma Jinyao" → ["Ma Jinyao", "Jinyao Ma"]
    """
    parts = name.strip().split()
    variants = [name.strip()]
    if len(parts) == 2:
        variants.append(f"{parts[1]} {parts[0]}")
    elif len(parts) >= 3:
        # 尝试把最后一个词作为姓提到前面
        variants.append(f"{parts[-1]} {' '.join(parts[:-1])}")
    # 去重保序
    seen = set()
    result = []
    for v in variants:
        if v.lower() not in seen:
            seen.add(v.lower())
            result.append(v)
    return result


def person_display_name(person: dict) -> str:
    """返回人员的规范显示名（中文名优先）"""
    return (person.get('chinese_name')
            or person.get('english_name')
            or person.get('pinyin', '?'))


def get_auto_threshold(query: str) -> int:
    """
    按查询串长度动态确定可接受的最大编辑距离:
      中文 2字 → 1, 3-4字 → 2, 5+字 → 3
      英文 按去空格后字符数的 1/4（至少 2）
    """
    if is_chinese(query):
        n = len(query)
        if n <= 2:
            return 1
        elif n <= 4:
            return 2
        else:
            return 3
    else:
        n = len(query.replace(' ', ''))
        return max(2, n // 4)


# ============================================================
# 3. 单人名称匹配
# ============================================================

def _distance_to_person(query: str, person: dict) -> tuple[int, str]:
    """
    计算 query 与一个 person 记录之间的最小编辑距离。
    同时考察 chinese_name / english_name / pinyin 三个字段。
    对英文查询自动尝试姓名顺序翻转。
    返回 (最小距离, 匹配到的字段名)
    """
    query_chinese = is_chinese(query)
    q_variants = [query] if query_chinese else english_name_variants(query)

    min_dist = 10 ** 6
    best_field = ''

    candidates: dict[str, list[str]] = {
        'chinese_name': [person.get('chinese_name', '')],
        'english_name': english_name_variants(person.get('english_name', '')),
        'pinyin':       english_name_variants(person.get('pinyin', '')),
    }

    # 中文查询时优先与中文名比较（权重不变，但若中文完全匹配则直接短路）
    field_order = (['chinese_name', 'pinyin', 'english_name']
                   if query_chinese
                   else ['english_name', 'pinyin', 'chinese_name'])

    for field in field_order:
        for fv in candidates[field]:
            if not fv:
                continue
            for qv in q_variants:
                d = levenshtein(qv, fv)
                if d < min_dist:
                    min_dist = d
                    best_field = field
                    if min_dist == 0:
                        return 0, field   # 完全匹配，提前退出

    return min_dist, best_field


def match_name(
    query: str,
    all_people: list[dict],
    global_threshold: Optional[int] = None
) -> dict:
    """
    将 query 与 all_people 中所有人做匹配。

    返回字典：
      status   : "exact" | "corrected" | "ambiguous" | "not_found"
      matched  : 匹配到的 person dict（ambiguous / not_found 时为 None）
      candidates: 候选列表（ambiguous 时为多个等距候选；not_found 时为最近前3供参考）
      distance : 最终编辑距离
      original : 原始查询串
      match_field: 匹配字段名
    """
    threshold = global_threshold if global_threshold is not None else get_auto_threshold(query)

    scored = []
    for person in all_people:
        dist, field = _distance_to_person(query, person)
        scored.append((dist, field, person))

    scored.sort(key=lambda x: x[0])

    if not scored:
        return dict(status='not_found', matched=None, candidates=[],
                    distance=9999, original=query, match_field='')

    min_dist = scored[0][0]
    best_group = [item for item in scored if item[0] == min_dist]

    # 距离超过阈值 → 视为未找到
    if min_dist > threshold:
        return dict(
            status='not_found',
            matched=None,
            candidates=[p for _, _, p in scored[:3]],   # 提供最近3个供人工参考
            distance=min_dist,
            original=query,
            match_field=''
        )

    if min_dist == 0:
        return dict(status='exact', matched=best_group[0][2], candidates=[],
                    distance=0, original=query, match_field=best_group[0][1])

    if len(best_group) == 1:
        return dict(status='corrected', matched=best_group[0][2], candidates=[],
                    distance=min_dist, original=query, match_field=best_group[0][1])

    # 多个等距候选 → 歧义
    return dict(status='ambiguous', matched=None,
                candidates=[p for _, _, p in best_group],
                distance=min_dist, original=query, match_field='')


# ============================================================
# 4. 单份证书校正
# ============================================================

def correct_certificate(cert_data: dict,
                         all_people: list[dict],
                         global_threshold: Optional[int] = None) -> dict:
    """
    对单份证书 JSON 进行 students / advisors 姓名校正。
    校正结果写回原字段；所有操作记录写入 _correction_meta。
    """
    result = deepcopy(cert_data)

    corrections_log: list[str] = []
    warnings_need_review: list[dict] = []
    removed_from_students: list[dict] = []
    removed_from_advisors: list[dict] = []

    # 最终输出列表
    out_students: list[str] = []
    out_advisors: list[str] = []

    # ── 4-A  处理 students ──────────────────────────────────
    for raw in (result.get('students') or []):
        raw = (raw or '').strip()
        if not raw:
            continue

        mr = match_name(raw, all_people, global_threshold)
        status = mr['status']

        if status == 'not_found':
            removed_from_students.append({
                'original_name': raw,
                'reason': (f"在 people_name.json 中未找到匹配项 "
                           f"(最近编辑距离={mr['distance']})"),
                'nearest_candidates': [person_display_name(p)
                                       for p in mr['candidates']]
            })
            corrections_log.append(
                f"[REMOVE-S] '{raw}' → 已从 students 移除（未找到匹配）"
            )

        elif status == 'ambiguous':
            cands = [person_display_name(p) for p in mr['candidates']]
            warnings_need_review.append({
                'field': 'students',
                'original_name': raw,
                'candidates': cands,
                'distance': mr['distance'],
                'message': (f"'{raw}' 存在 {len(cands)} 个等距候选，"
                            f"请人工确认: {cands}")
            })
            corrections_log.append(
                f"[AMBIGUOUS-S] '{raw}' → 歧义候选: {cands}，等待人工确认"
            )
            # 不加入 out_students，保留在 warnings 中等人工处理

        else:  # exact or corrected
            person = mr['matched']
            display = person_display_name(person)
            group = person['_group']

            if raw != display:
                corrections_log.append(
                    f"[CORRECT-S] '{raw}' → '{display}' "
                    f"(距离={mr['distance']}, 匹配字段={mr['match_field']})"
                )

            if group == 'students':
                if display not in out_students:
                    out_students.append(display)

            elif group == 'advisors':
                # 该人在 people_name.json 中属于 advisors，移至 advisors
                if display not in out_advisors:
                    out_advisors.append(display)
                corrections_log.append(
                    f"[MOVE] '{raw}' (→'{display}') 在 people_name.json 中"
                    f"属于 advisors，已从 students 移至 advisors"
                )
                warnings_need_review.append({
                    'field': 'students→advisors',
                    'original_name': raw,
                    'corrected_name': display,
                    'message': (f"'{raw}' 在 people_name.json 中属于 advisors，"
                                f"已自动移至 advisors 字段")
                })

    # ── 4-B  处理 advisors ─────────────────────────────────
    for raw in (result.get('advisors') or []):
        raw = (raw or '').strip()
        if not raw:
            continue

        mr = match_name(raw, all_people, global_threshold)
        status = mr['status']

        if status == 'not_found':
            removed_from_advisors.append({
                'original_name': raw,
                'reason': (f"在 people_name.json 中未找到匹配项 "
                           f"(最近编辑距离={mr['distance']})"),
                'nearest_candidates': [person_display_name(p)
                                       for p in mr['candidates']]
            })
            corrections_log.append(
                f"[REMOVE-A] '{raw}' → 已从 advisors 移除（未找到匹配）"
            )

        elif status == 'ambiguous':
            cands = [person_display_name(p) for p in mr['candidates']]
            warnings_need_review.append({
                'field': 'advisors',
                'original_name': raw,
                'candidates': cands,
                'distance': mr['distance'],
                'message': (f"'{raw}' 存在 {len(cands)} 个等距候选，"
                            f"请人工确认: {cands}")
            })
            corrections_log.append(
                f"[AMBIGUOUS-A] '{raw}' → 歧义候选: {cands}，等待人工确认"
            )

        else:  # exact or corrected
            person = mr['matched']
            display = person_display_name(person)

            if raw != display:
                corrections_log.append(
                    f"[CORRECT-A] '{raw}' → '{display}' "
                    f"(距离={mr['distance']}, 匹配字段={mr['match_field']})"
                )

            if display not in out_advisors:
                out_advisors.append(display)

    # ── 4-C  写回结果 ──────────────────────────────────────
    result['students'] = out_students
    result['advisors'] = out_advisors

    # 写入校正元信息（仅在有内容时添加该字段）
    meta: dict = {}
    if corrections_log:
        meta['corrections_log'] = corrections_log
    if warnings_need_review:
        meta['warnings_need_review'] = warnings_need_review
    if removed_from_students:
        meta['removed_from_students'] = removed_from_students
    if removed_from_advisors:
        meta['removed_from_advisors'] = removed_from_advisors

    if meta:
        result['_correction_meta'] = meta
    elif '_correction_meta' in result:
        del result['_correction_meta']

    return result


# ============================================================
# 5. 批量处理入口
# ============================================================

def setup_logger() -> logging.Logger:
    logger = logging.getLogger('correct_names')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'
    ))
    logger.addHandler(handler)
    return logger


def main():
    parser = argparse.ArgumentParser(
        description='Stage 3 - 证书姓名校正工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_dir',
        default='/Users/zhongzuodong/AutoCert/results/qwen_extract',
        help='Stage 2 输出的 .json 目录'
    )
    parser.add_argument(
        '--output_dir',
        default='/Users/zhongzuodong/AutoCert/results/corrected',
        help='校正后 .json 的保存目录'
    )
    parser.add_argument(
        '--people_file',
        default='/Users/zhongzuodong/AutoCert/roster/people_name.json',
        help='people_name.json 路径'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=None,
        help='全局最大编辑距离阈值（不填则按姓名长度自动决定）'
    )
    args = parser.parse_args()

    logger = setup_logger()

    # ── 加载 people_name.json ──────────────────────────────
    if not os.path.exists(args.people_file):
        logger.error(f"people_name.json 不存在: {args.people_file}")
        sys.exit(1)

    with open(args.people_file, 'r', encoding='utf-8') as f:
        people_raw = json.load(f)

    all_people: list[dict] = []
    for p in people_raw.get('students', []):
        dp = dict(p); dp['_group'] = 'students'
        all_people.append(dp)
    for p in people_raw.get('advisors', []):
        dp = dict(p); dp['_group'] = 'advisors'
        all_people.append(dp)

    n_stu = sum(1 for p in all_people if p['_group'] == 'students')
    n_adv = sum(1 for p in all_people if p['_group'] == 'advisors')
    logger.info(f"people_name.json 加载完成：{n_stu} 名学生 / {n_adv} 名教师")

    # ── 准备目录 ───────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    input_path = Path(args.input_dir)
    json_files = sorted(input_path.glob('*.json'))

    if not json_files:
        logger.warning(f"在 {args.input_dir} 中未找到 .json 文件")
        return

    logger.info(f"共找到 {len(json_files)} 个 .json 文件，开始校正...")

    # ── 逐文件处理 ─────────────────────────────────────────
    ok, err = 0, 0
    sep = '─' * 56

    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                cert = json.load(f)

            corrected = correct_certificate(cert, all_people, args.threshold)

            out_path = Path(args.output_dir) / jf.name
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(corrected, f, ensure_ascii=False, indent=2)

            # 打印摘要
            meta = corrected.get('_correction_meta', {})
            logs  = meta.get('corrections_log', [])
            warns = meta.get('warnings_need_review', [])
            rm_s  = meta.get('removed_from_students', [])
            rm_a  = meta.get('removed_from_advisors', [])

            logger.info(sep)
            logger.info(f"✅  {jf.name}")
            if not meta:
                logger.info("    (无需校正，全部精确匹配)")
            for log_line in logs:
                logger.info(f"    {log_line}")
            for w in warns:
                logger.warning(f"    ⚠️  {w['message']}")
            for r in rm_s:
                logger.warning(
                    f"    ❌ 从 students 移除: '{r['original_name']}' "
                    f"| {r['reason']} "
                    f"| 最近候选: {r['nearest_candidates']}"
                )
            for r in rm_a:
                logger.warning(
                    f"    ❌ 从 advisors 移除: '{r['original_name']}' "
                    f"| {r['reason']} "
                    f"| 最近候选: {r['nearest_candidates']}"
                )

            ok += 1

        except Exception as exc:
            logger.error(f"❌ 处理 {jf.name} 时出错: {exc}", exc_info=True)
            err += 1

    # ── 总结 ──────────────────────────────────────────────
    logger.info(sep)
    logger.info(f"完成！成功 {ok} 个 / 失败 {err} 个")
    logger.info(f"校正结果已保存至: {args.output_dir}")


if __name__ == '__main__':
    main()
