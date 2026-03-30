#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_score_allocation.py
验证学生提交的分数分配表，检查分配合理性，输出最终确认分数或问题报告
"""

import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from collections import defaultdict
from dataclasses import dataclass, field

# ─── 路径配置 ────────────────────────────────────────────────────────────────

BASE_DIR = Path("/mnt/tidal-alsh01/dataset/OCRData/page_label_project/AutoCert")
SYSTEM_RESULTS_DIR = BASE_DIR / "results" / "students_score"
STUDENT_FORMS_DIR = BASE_DIR / "results" / "student_forms"
VALIDATION_OUTPUT_DIR = BASE_DIR / "results" / "validation_results"
VALIDATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 数据结构定义
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationIssue:
    """验证问题记录"""
    issue_type: str           # error / warning / info
    category: str             # 问题分类
    award_name: str           # 相关奖项
    message: str              # 问题描述
    expected: Any = None      # 期望值
    actual: Any = None        # 实际值
    suggestion: str = ""      # 修改建议
    
    def to_dict(self) -> dict:
        return {
            "issue_type": self.issue_type,
            "category": self.category,
            "award_name": self.award_name,
            "message": self.message,
            "expected": self.expected,
            "actual": self.actual,
            "suggestion": self.suggestion,
        }


@dataclass
class AwardValidationResult:
    """单个奖项的验证结果"""
    system_award_name: Optional[str]
    student_award_name: Optional[str]
    match_similarity: float
    is_matched: bool
    is_valid: bool
    confirmed_score: Optional[float]
    # 新增：系统计算的分数区间
    system_score_min: Optional[float] = None
    system_score_max: Optional[float] = None
    system_team_total: Optional[float] = None
    system_individual_max: Optional[float] = None
    issues: List[ValidationIssue] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "system_award_name": self.system_award_name,
            "student_award_name": self.student_award_name,
            "match_similarity": round(self.match_similarity, 4),
            "is_matched": self.is_matched,
            "is_valid": self.is_valid,
            "confirmed_score": self.confirmed_score,
            # 新增：系统分数信息
            "system_scoring": {
                "score_min": self.system_score_min,
                "score_max": self.system_score_max,
                "team_total": self.system_team_total,
                "individual_max": self.system_individual_max,
            },
            "issues": [i.to_dict() for i in self.issues],
        }



# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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


def preprocess_award_name(s: str) -> str:
    """预处理奖项名称用于匹配"""
    s = re.sub(r'\d{4}(?:年|—|-)?', '', s)          # 移除年份
    s = re.sub(r'第[\d一二三四五六七八九十]+届', '', s)  # 移除"第X届"
    s = re.sub(r'[（）()\[\]【】「」\s\"\'""''·]', '', s)  # 移除括号和空白
    return s.strip().lower()


def extract_year(s: str) -> Optional[str]:
    """从字符串中提取年份"""
    if not s:
        return None
    match = re.search(r'(\d{4})', str(s))
    return match.group(1) if match else None


def is_float_equal(a: Optional[float], b: Optional[float], tolerance: float = 0.01) -> bool:
    """浮点数相等比较（允许误差）"""
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tolerance


# ══════════════════════════════════════════════════════════════════════════════
# 奖项匹配引擎
# ══════════════════════════════════════════════════════════════════════════════

class AwardMatcher:
    """奖项匹配器：将学生提交的奖项与系统识别的奖项进行配对"""
    
    MATCH_THRESHOLD = 0.45  # 相似度阈值
    
    def __init__(self, system_entries: List[Dict], student_awards: List[Dict]):
        self.system_entries = system_entries
        self.student_awards = student_awards
    
    def match(self) -> List[Dict]:
        """执行匹配，返回配对结果"""
        pairs = []
        used_system = set()
        used_student = set()
        
        # 计算所有可能的配对分数
        candidates = []
        for i, sys_entry in enumerate(self.system_entries):
            sys_name = preprocess_award_name(sys_entry.get("certificate_name", ""))
            sys_year = extract_year(sys_entry.get("issue_date", ""))
            
            for j, stu_award in enumerate(self.student_awards):
                stu_name = preprocess_award_name(stu_award.get("award_name", ""))
                stu_year = extract_year(str(stu_award.get("year", "")))
                
                # 计算名称相似度
                sim = similarity_ratio(sys_name, stu_name)
                
                # 年份匹配加分
                if sys_year and stu_year and sys_year == stu_year:
                    sim = min(1.0, sim + 0.15)
                
                if sim >= self.MATCH_THRESHOLD:
                    candidates.append({
                        "sys_idx": i,
                        "stu_idx": j,
                        "similarity": sim,
                    })
        
        # 贪心匹配：按相似度从高到低选择
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        
        for cand in candidates:
            i, j = cand["sys_idx"], cand["stu_idx"]
            if i not in used_system and j not in used_student:
                pairs.append({
                    "system_entry": self.system_entries[i],
                    "student_award": self.student_awards[j],
                    "similarity": cand["similarity"],
                    "sys_idx": i,
                    "stu_idx": j,
                })
                used_system.add(i)
                used_student.add(j)
        
        # 添加未匹配的系统记录
        for i, sys_entry in enumerate(self.system_entries):
            if i not in used_system:
                pairs.append({
                    "system_entry": sys_entry,
                    "student_award": None,
                    "similarity": 0.0,
                    "sys_idx": i,
                    "stu_idx": -1,
                })
        
        # 添加未匹配的学生提交
        for j, stu_award in enumerate(self.student_awards):
            if j not in used_student:
                pairs.append({
                    "system_entry": None,
                    "student_award": stu_award,
                    "similarity": 0.0,
                    "sys_idx": -1,
                    "stu_idx": j,
                })
        
        return pairs


# ══════════════════════════════════════════════════════════════════════════════
# 核心验证逻辑
# ══════════════════════════════════════════════════════════════════════════════

class ScoreAllocationValidator:
    """分数分配验证器"""
    
    def __init__(self, system_result: Dict, student_form: Dict):
        self.system_result = system_result
        self.student_form = student_form
        self.student_name = student_form.get("student_name", "")
        self.issues: List[ValidationIssue] = []
        self.award_results: List[AwardValidationResult] = []
    
    def validate(self) -> Dict:
        """执行完整验证流程"""
        
        # 1. 验证学生姓名一致性
        self._validate_student_name()
        
        # 2. 匹配奖项
        system_entries = self.system_result.get("score_details", [])
        student_awards = self.student_form.get("awards", [])
        
        matcher = AwardMatcher(system_entries, student_awards)
        matched_pairs = matcher.match()
        
        # 3. 逐个验证奖项
        confirmed_scores = []
        
        for pair in matched_pairs:
            result = self._validate_award_pair(pair)
            self.award_results.append(result)
            
            if result.is_valid and result.confirmed_score is not None:
                confirmed_scores.append({
                    "award_name": result.system_award_name or result.student_award_name,
                    "score": result.confirmed_score,
                })
        
        # 4. 应用系统的去重和上限规则
        final_scores = self._apply_system_rules(confirmed_scores)
        
        # 5. 生成最终报告
        return self._generate_report(final_scores)
    
    def _validate_student_name(self):
        """验证学生姓名"""
        system_name = self.system_result.get("student_name", "")
        if self.student_name != system_name:
            self.issues.append(ValidationIssue(
                issue_type="error",
                category="姓名不一致",
                award_name="全局",
                message=f"表单姓名与系统记录不一致",
                expected=system_name,
                actual=self.student_name,
                suggestion="请确认提交的表单是否属于本人"
            ))
    
    def _validate_award_pair(self, pair: Dict) -> AwardValidationResult:
        """验证单个奖项配对"""
        sys_entry = pair.get("system_entry")
        stu_award = pair.get("student_award")
        similarity = pair.get("similarity", 0.0)
        
        # 初始化结果
        result = AwardValidationResult(
            system_award_name=sys_entry.get("certificate_name") if sys_entry else None,
            student_award_name=stu_award.get("award_name") if stu_award else None,
            match_similarity=similarity,
            is_matched=sys_entry is not None and stu_award is not None,
            is_valid=False,
            confirmed_score=None,
        )
        
        # ─── 情况1：系统有记录但学生未提交 ────────────────────────────
        if sys_entry and not stu_award:
            # 检查是否是被系统跳过的奖项
            skipped = self.system_result.get("score_summary", {}).get("skipped_entries", [])
            is_skipped = any(
                s.get("certificate_name") == sys_entry.get("certificate_name")
                for s in skipped
            )
            
            if is_skipped:
                result.issues.append(ValidationIssue(
                    issue_type="info",
                    category="奖项未提交",
                    award_name=sys_entry.get("certificate_name", ""),
                    message="该奖项已被系统跳过（可能因规则缺失或重复），无需提交",
                ))
                result.is_valid = True
            else:
                result.issues.append(ValidationIssue(
                    issue_type="warning",
                    category="奖项未提交",
                    award_name=sys_entry.get("certificate_name", ""),
                    message="系统识别到的奖项未在提交表单中找到",
                    suggestion="请确认是否遗漏此奖项，或确认此奖项不需要加分"
                ))
                # 警告级别，不影响其他奖项验证
                result.is_valid = True
            return result
        
        # ─── 情况2：学生提交了但系统没有记录 ────────────────────────────
        if stu_award and not sys_entry:
            result.issues.append(ValidationIssue(
                issue_type="error",
                category="奖项不存在",
                award_name=stu_award.get("award_name", ""),
                message="提交的奖项在系统证书记录中未找到",
                suggestion="请确认证书已正确上传并被系统识别，或检查奖项名称是否填写正确"
            ))
            return result
        
        # ─── 情况3：两边都有，进行详细验证 ────────────────────────────────
        if sys_entry and stu_award:
            return self._validate_matched_award(sys_entry, stu_award, result)
        
        return result
    
    def _validate_matched_award(
        self, 
        sys_entry: Dict, 
        stu_award: Dict, 
        result: AwardValidationResult
    ) -> AwardValidationResult:
        """验证已匹配的奖项对"""
        
        award_name = sys_entry.get("certificate_name", "")
        
        # ─── 检查1：团队/个人类型一致性 ────────────────────────────────
        is_team_sys = sys_entry.get("matched_rule", {}).get("is_team", False)
        is_team_stu = stu_award.get("is_team", False)
        
        if is_team_sys != is_team_stu:
            result.issues.append(ValidationIssue(
                issue_type="error",
                category="类型不匹配",
                award_name=award_name,
                message=f"比赛类型不一致",
                expected="团队项目" if is_team_sys else "个人项目",
                actual="团队项目" if is_team_stu else "个人项目",
            ))
            return result
        
        # ─── 检查2：年份一致性 ────────────────────────────────────────
        sys_year = extract_year(sys_entry.get("issue_date", ""))
        stu_year = extract_year(str(stu_award.get("year", "")))
        
        if sys_year and stu_year and sys_year != stu_year:
            result.issues.append(ValidationIssue(
                issue_type="warning",
                category="年份不一致",
                award_name=award_name,
                message="年份信息不一致，请确认",
                expected=sys_year,
                actual=stu_year,
            ))
        
        # ─── 检查3：奖项等级一致性 ────────────────────────────────────
        sys_grade = sys_entry.get("award_level_normalized")
        stu_grade = self._normalize_grade(stu_award.get("grade"))
        
        if sys_grade and stu_grade and sys_grade != stu_grade:
            result.issues.append(ValidationIssue(
                issue_type="error",
                category="奖项等级不一致",
                award_name=award_name,
                message="奖项等级与系统识别不一致",
                expected=sys_grade,
                actual=stu_grade,
                suggestion="请核对证书原件上的奖项等级"
            ))
            return result
        
        # ─── 分流验证：团队赛 vs 个人赛 ────────────────────────────────
        if is_team_sys:
            return self._validate_team_award(sys_entry, stu_award, result)
        else:
            return self._validate_individual_award(sys_entry, stu_award, result)
    
    def _normalize_grade(self, grade: Optional[str]) -> Optional[str]:
        """归一化奖项等级"""
        if not grade:
            return None
        mapping = {
            "一等奖": "一等", "二等奖": "二等", "三等奖": "三等",
            "一等": "一等", "二等": "二等", "三等": "三等",
            "金奖": "一等", "银奖": "二等", "铜奖": "三等",
            "特等奖": "特等", "优秀奖": "优秀",
            "无": None, "": None,
        }
        return mapping.get(grade.strip(), grade.strip())
    
    def _validate_team_award(
        self, 
        sys_entry: Dict, 
        stu_award: Dict, 
        result: AwardValidationResult
    ) -> AwardValidationResult:
        """验证团队赛分数分配"""
        
        award_name = sys_entry.get("certificate_name", "")
        members = stu_award.get("members", [])
        scoring = sys_entry.get("scoring", {})
        
        team_total = scoring.get("team_total")
        individual_max = scoring.get("individual_max")
        score_range_min = scoring.get("score_range_min")
        score_range_max = scoring.get("score_range_max")

        # ═══════════════════════════════════════════════════════════════
        # 【新增】记录系统计算的分数区间
        # ═══════════════════════════════════════════════════════════════
        result.system_score_min = score_range_min
        result.system_score_max = score_range_max
        result.system_team_total = team_total
        result.system_individual_max = individual_max
        
        all_checks_passed = True
        
        # ─── 检查1：成员列表不能为空 ────────────────────────────────────
        if not members:
            result.issues.append(ValidationIssue(
                issue_type="error",
                category="成员列表为空",
                award_name=award_name,
                message="团队成员及分数分配列表为空",
                suggestion="请填写完整的团队成员及对应分数"
            ))
            return result
        
        # ─── 检查2：团队总分是否超限 ────────────────────────────────────
        total_allocated = sum(m.get("points", 0) for m in members)

        if team_total is not None and total_allocated > team_total + 0.001:  # 允许微小误差
            result.issues.append(ValidationIssue(
                issue_type="error",
                category="总分超限",
                award_name=award_name,
                message="团队分数分配总和超过规定团队总分",
                expected=f"≤ {team_total}",
                actual=round(total_allocated, 4),
                suggestion=f"所有成员分数之和不能超过 {team_total}"
            ))
            all_checks_passed = False
        elif team_total is not None and total_allocated < team_total - 0.001:
            # 可选：添加一个提示（非错误），告知未用完全部分数
            result.issues.append(ValidationIssue(
                issue_type="info",
                category="总分未用完",
                award_name=award_name,
                message=f"团队分数未完全分配（剩余 {round(team_total - total_allocated, 4)} 分）",
                expected=team_total,
                actual=round(total_allocated, 4),
                suggestion="这是允许的，但建议确认是否故意放弃部分加分"
            ))

        
        # ─── 检查3：排名顺序（前面的分数 >= 后面的分数）────────────────
        points_list = [m.get("points", 0) for m in members]
        for k in range(len(points_list) - 1):
            if points_list[k] < points_list[k + 1] - 0.001:  # 允许微小误差
                result.issues.append(ValidationIssue(
                    issue_type="error",
                    category="排名顺序违规",
                    award_name=award_name,
                    message=f"第{k+1}名的分数小于第{k+2}名",
                    expected=f"第{k+1}名({members[k]['name']})分数 ≥ 第{k+2}名({members[k+1]['name']})分数",
                    actual=f"{points_list[k]} < {points_list[k+1]}",
                    suggestion="按照排名顺序，靠前成员的分数应大于等于靠后成员"
                ))
                all_checks_passed = False
        
        # ─── 检查4：个人分数不超过上限 ──────────────────────────────────
        if individual_max is not None:
            for m in members:
                if m.get("points", 0) > individual_max + 0.001:
                    result.issues.append(ValidationIssue(
                        issue_type="error",
                        category="超过个人上限",
                        award_name=award_name,
                        message=f"成员 {m['name']} 的分数超过个人上限",
                        expected=f"≤ {individual_max}",
                        actual=m['points'],
                        suggestion=f"每位成员的分数不能超过 {individual_max} 分"
                    ))
                    all_checks_passed = False
        
        # ─── 检查5：所有分数非负 ────────────────────────────────────────
        for m in members:
            if m.get("points", 0) < 0:
                result.issues.append(ValidationIssue(
                    issue_type="error",
                    category="分数为负",
                    award_name=award_name,
                    message=f"成员 {m['name']} 的分数为负数",
                    actual=m['points'],
                    suggestion="分数不能为负数"
                ))
                all_checks_passed = False
        
        # ─── 检查6：验证成员列表与证书一致 ──────────────────────────────
        system_students = sys_entry.get("students_in_cert", [])
        submitted_names = [m.get("name", "") for m in members]
        
        system_set = set(system_students)
        submitted_set = set(submitted_names)
        
        if system_set != submitted_set:
            missing = system_set - submitted_set
            extra = submitted_set - system_set
            
            if missing:
                result.issues.append(ValidationIssue(
                    issue_type="warning",
                    category="成员列表不完整",
                    award_name=award_name,
                    message=f"提交的成员列表缺少以下人员",
                    expected=list(system_set),
                    actual=list(submitted_set),
                    suggestion=f"缺少: {list(missing)}"
                ))
            
            if extra:
                result.issues.append(ValidationIssue(
                    issue_type="warning",
                    category="成员列表有多余",
                    award_name=award_name,
                    message=f"提交的成员列表包含证书中不存在的人员",
                    expected=list(system_set),
                    actual=list(submitted_set),
                    suggestion=f"多余: {list(extra)}"
                ))
        
        # ─── 检查7：当前学生的分数是否在合法区间内 ────────────────────────
        student_score = None
        for m in members:
            if m.get("name") == self.student_name:
                student_score = m.get("points", 0)
                break
        
        if student_score is None:
            result.issues.append(ValidationIssue(
                issue_type="error",
                category="未找到本人",
                award_name=award_name,
                message=f"在提交的成员列表中未找到学生 {self.student_name}",
                suggestion="请确保成员列表中包含本人"
            ))
            all_checks_passed = False
        else:
            if score_range_min is not None and student_score < score_range_min - 0.001:
                result.issues.append(ValidationIssue(
                    issue_type="error",
                    category="分数低于最低值",
                    award_name=award_name,
                    message=f"您的分数低于系统计算的合法最低分",
                    expected=f"≥ {score_range_min}",
                    actual=student_score,
                    suggestion=f"根据您的排名，分数应不低于 {score_range_min}"
                ))
                all_checks_passed = False
            
            if score_range_max is not None and student_score > score_range_max + 0.001:
                result.issues.append(ValidationIssue(
                    issue_type="error",
                    category="分数高于最高值",
                    award_name=award_name,
                    message=f"您的分数高于系统计算的合法最高分",
                    expected=f"≤ {score_range_max}",
                    actual=student_score,
                    suggestion=f"根据您的排名，分数应不高于 {score_range_max}"
                ))
                all_checks_passed = False
        
        # ─── 检查8：排名顺序与证书一致 ────────────────────────────────────
        stu_rank = stu_award.get("team_rank")
        sys_rank = sys_entry.get("student_rank_in_team")
        
        if stu_rank and sys_rank and stu_rank != sys_rank:
            result.issues.append(ValidationIssue(
                issue_type="warning",
                category="排名不一致",
                award_name=award_name,
                message=f"填写的排名与系统识别不一致",
                expected=sys_rank,
                actual=stu_rank,
                suggestion="请核对证书上的排名顺序"
            ))
        
        # ─── 汇总结果 ────────────────────────────────────────────────────
        result.is_valid = all_checks_passed
        if all_checks_passed and student_score is not None:
            result.confirmed_score = student_score
        
        return result
    
    def _validate_individual_award(
        self, 
        sys_entry: Dict, 
        stu_award: Dict, 
        result: AwardValidationResult
    ) -> AwardValidationResult:
        """验证个人赛分数"""
        
        award_name = sys_entry.get("certificate_name", "")
        scoring = sys_entry.get("scoring", {})
        
        system_score = scoring.get("score_fixed") or scoring.get("individual_max")
        student_score = stu_award.get("bonus_points")

        # ═══════════════════════════════════════════════════════════════
        # 【新增】记录系统计算的分数（个人赛 min=max）
        # ═══════════════════════════════════════════════════════════════
        result.system_score_min = system_score
        result.system_score_max = system_score
        result.system_individual_max = scoring.get("individual_max")
        result.system_team_total = None  # 个人赛无团队总分
        
        # ─── 检查1：系统分数是否存在 ────────────────────────────────────
        if system_score is None:
            result.issues.append(ValidationIssue(
                issue_type="error",
                category="系统计算失败",
                award_name=award_name,
                message="系统未能计算出该奖项的分数",
                suggestion="请联系管理员检查加分规则配置"
            ))
            return result
        
        # ─── 检查2：学生是否填写分数 ────────────────────────────────────
        if student_score is None:
            result.issues.append(ValidationIssue(
                issue_type="error",
                category="分数未填写",
                award_name=award_name,
                message="未填写该奖项的加分",
                expected=system_score,
                suggestion=f"请填写加分，应为 {system_score}"
            ))
            return result
        
        # ─── 检查3：分数是否一致 ────────────────────────────────────────
        if not is_float_equal(system_score, student_score):
            result.issues.append(ValidationIssue(
                issue_type="error",
                category="分数不一致",
                award_name=award_name,
                message="填写的分数与系统计算不一致",
                expected=system_score,
                actual=student_score,
                suggestion=f"根据规则，该奖项应加 {system_score} 分"
            ))
            return result
        
        result.is_valid = True
        result.confirmed_score = system_score
        return result
    
    def _apply_system_rules(self, confirmed_scores: List[Dict]) -> Dict:
        """应用系统的去重和上限规则"""
        
        # 参照系统已有的去重结果
        system_summary = self.system_result.get("score_summary", {})
        effective_entries = system_summary.get("effective_entries", [])
        
        # 构建有效奖项映射
        effective_map = {
            e.get("certificate_name"): e for e in effective_entries
        }
        
        final_scores = []
        dedup_notes = []
        cap_notes = []
        
        for cs in confirmed_scores:
            award_name = cs["award_name"]
            
            # 检查是否在有效列表中
            if award_name in effective_map:
                eff = effective_map[award_name]
                # 使用系统计算的上限后分数（如果适用）
                final_score = cs["score"]
                
                # 检查类别上限
                if eff.get("score_max") != eff.get("score_min"):
                    # 团队项目，使用学生分配的分数
                    final_score = cs["score"]
                else:
                    final_score = eff.get("score_max", cs["score"])
                
                final_scores.append({
                    "award_name": award_name,
                    "original_score": cs["score"],
                    "final_score": final_score,
                    "category": eff.get("category"),
                    "year": eff.get("year"),
                })
            else:
                # 可能被去重或上限裁剪
                dedup_notes.append({
                    "award_name": award_name,
                    "note": "该奖项可能因同年同类别去重或上限规则被跳过"
                })
        
        # 计算总分
        total = sum(fs["final_score"] for fs in final_scores)
        
        return {
            "final_scores": final_scores,
            "total": round(total, 4),
            "dedup_notes": dedup_notes,
            "cap_notes": cap_notes,
        }
    
    def _generate_report(self, final_scores: Dict) -> Dict:
        """生成最终验证报告"""
        
        # 统计问题
        error_count = sum(
            1 for ar in self.award_results 
            for issue in ar.issues 
            if issue.issue_type == "error"
        )
        warning_count = sum(
            1 for ar in self.award_results 
            for issue in ar.issues 
            if issue.issue_type == "warning"
        )
        
        # 加上全局问题
        error_count += sum(1 for i in self.issues if i.issue_type == "error")
        warning_count += sum(1 for i in self.issues if i.issue_type == "warning")
        
        # 判断整体状态
        if error_count > 0:
            overall_status = "failed"
        elif warning_count > 0:
            overall_status = "passed_with_warnings"
        else:
            overall_status = "passed"

        # ═══════════════════════════════════════════════════════════════
        # 【新增】构建包含系统分数区间的分数明细
        # ═══════════════════════════════════════════════════════════════
        score_breakdown_detailed = []
        for fs in final_scores.get("final_scores", []):
            award_name = fs["award_name"]
            
            # 从 award_results 中找到对应的系统分数信息
            matching_result = next(
                (ar for ar in self.award_results 
                if ar.system_award_name == award_name),
                None
            )
            
            detail = {
                "award_name": award_name,
                "original_score": fs.get("original_score"),
                "final_score": fs.get("final_score"),
                "category": fs.get("category"),
                "year": fs.get("year"),
                # 新增：系统分数区间
                "system_score_min": matching_result.system_score_min if matching_result else None,
                "system_score_max": matching_result.system_score_max if matching_result else None,
                "system_team_total": matching_result.system_team_total if matching_result else None,
                "system_individual_max": matching_result.system_individual_max if matching_result else None,
            }
            score_breakdown_detailed.append(detail)
        
        return {
            "student_name": self.student_name,
            "validation_time": datetime.now().isoformat(timespec="seconds"),
            "overall_status": overall_status,
            "error_count": error_count,
            "warning_count": warning_count,
            "confirmed_total_score": final_scores["total"] if overall_status != "failed" else None,
            "score_breakdown": final_scores["final_scores"] if overall_status != "failed" else [],
            "global_issues": [i.to_dict() for i in self.issues],
            "award_validations": [ar.to_dict() for ar in self.award_results],
            "dedup_notes": final_scores.get("dedup_notes", []),
            "system_warnings": self.system_result.get("global_rules_reminder", []),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 报告输出
# ══════════════════════════════════════════════════════════════════════════════

def print_validation_report(result: Dict):
    """打印验证报告到控制台"""
    
    status = result["overall_status"]
    student = result["student_name"]
    
    print(f"\n{'═'*70}")
    print(f"  📋 分数分配验证报告")
    print(f"  学生姓名：{student}")
    print(f"  验证时间：{result['validation_time']}")
    print(f"{'═'*70}\n")
    
    # ─── 状态总览 ────────────────────────────────────────────────────────
    if status == "passed":
        print(f"  ✅ 验证通过！\n")
    elif status == "passed_with_warnings":
        print(f"  ⚠️  验证通过（存在 {result['warning_count']} 个警告需关注）\n")
    else:
        print(f"  ❌ 验证失败（{result['error_count']} 个错误，{result['warning_count']} 个警告）\n")
    
    # ─── 确认分数 ────────────────────────────────────────────────────────
    if result["confirmed_total_score"] is not None:
        print(f"  📊 确认总分：{result['confirmed_total_score']} 分")
        print(f"\n  分数明细：")
        print(f"  {'─'*60}")
        for item in result["score_breakdown"]:
            print(f"  • {item['award_name']}")
            print(f"    └─ {item['final_score']} 分 (类别: {item.get('category', 'N/A')})")
        print(f"  {'─'*60}")
    
    # ─── 问题列表 ────────────────────────────────────────────────────────
    if result["error_count"] > 0 or result["warning_count"] > 0:
        print(f"\n  📝 问题详情：")
        print(f"  {'─'*60}")
        
        # 全局问题
        for issue in result["global_issues"]:
            icon = "❌" if issue["issue_type"] == "error" else "⚠️"
            print(f"\n  {icon} [{issue['category']}]")
            print(f"     {issue['message']}")
            if issue.get("expected"):
                print(f"     期望: {issue['expected']}")
            if issue.get("actual"):
                print(f"     实际: {issue['actual']}")
            if issue.get("suggestion"):
                print(f"     💡 建议: {issue['suggestion']}")
        
        # 各奖项问题
        for av in result["award_validations"]:
            if av["issues"]:
                award_name = av["system_award_name"] or av["student_award_name"]
                print(f"\n  📄 奖项: {award_name}")
                
                for issue in av["issues"]:
                    icon = "❌" if issue["issue_type"] == "error" else "⚠️" if issue["issue_type"] == "warning" else "ℹ️"
                    print(f"     {icon} [{issue['category']}]")
                    print(f"        {issue['message']}")
                    if issue.get("expected"):
                        print(f"        期望: {issue['expected']}")
                    if issue.get("actual"):
                        print(f"        实际: {issue['actual']}")
                    if issue.get("suggestion"):
                        print(f"        💡 建议: {issue['suggestion']}")
    
    # ─── 结论 ────────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    if status == "failed":
        print(f"  ⚠️  请修正上述错误后重新提交，或联系管理员人工核验")
    elif status == "passed_with_warnings":
        print(f"  ✅ 可以使用确认分数，但建议关注上述警告信息")
    else:
        print(f"  ✅ 所有验证通过，分数已确认")
    print(f"{'═'*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="验证学生分数分配表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python validate_score_allocation.py -s system_result.json -f student_form.json
  python validate_score_allocation.py -s system_result.json -f student_form.json -o output.json
        """
    )
    parser.add_argument("--system-result", "-s", required=True,
                        help="系统计算结果 JSON 文件路径")
    parser.add_argument("--student-form", "-f", required=True,
                        help="学生提交的分配表 JSON 文件路径")
    parser.add_argument("--output", "-o",
                        help="输出文件路径（可选，默认自动生成）")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="静默模式，只输出 JSON 不打印报告")
    
    args = parser.parse_args()
    
    # 加载文件
    try:
        system_result = load_json(Path(args.system_result))
        student_form = load_json(Path(args.student_form))
    except FileNotFoundError as e:
        print(f"错误：找不到文件 - {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"错误：JSON 解析失败 - {e}")
        return 1
    
    # 执行验证
    validator = ScoreAllocationValidator(system_result, student_form)
    result = validator.validate()
    
    # 打印报告
    if not args.quiet:
        print_validation_report(result)
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        student_name = result["student_name"]
        output_path = VALIDATION_OUTPUT_DIR / f"{student_name}_validation_{timestamp}.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    if not args.quiet:
        print(f"  📄 详细报告已保存至：{output_path}\n")
    
    # 返回状态码
    return 0 if result["overall_status"] != "failed" else 1


if __name__ == "__main__":
    exit(main())
