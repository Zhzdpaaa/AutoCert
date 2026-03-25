#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
探查加分规则 Excel 文件的原始结构
运行后将全部内容打印到终端，同时保存为 inspect_output.txt
用法:
    python tools/inspect_rules_xls.py
"""

import sys
import os

# ── 依赖检测 ──────────────────────────────────────────────
try:
    import xlrd
    _ENGINE = 'xlrd'
except ImportError:
    xlrd = None

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

# ── 配置 ──────────────────────────────────────────────────
XLS_PATH = (
    "/Users/zhongzuodong/AutoCert/roster/"
    "附件1：北京科技大学自动化学院2026届本科毕业生保研加分规则.xls"
)
OUTPUT_TXT = "/Users/zhongzuodong/AutoCert/roster/inspect_output.txt"

SEP_MAJOR = "=" * 70
SEP_MINOR = "-" * 50


def cell_str(cell) -> str:
    """将 xlrd Cell 转为可读字符串"""
    import xlrd as _xlrd
    if cell.ctype == _xlrd.XL_CELL_EMPTY:
        return ""
    if cell.ctype == _xlrd.XL_CELL_NUMBER:
        v = cell.value
        # 整数显示为整数
        return str(int(v)) if v == int(v) else str(v)
    if cell.ctype == _xlrd.XL_CELL_DATE:
        return f"[DATE:{cell.value}]"
    if cell.ctype == _xlrd.XL_CELL_BOOLEAN:
        return "TRUE" if cell.value else "FALSE"
    return str(cell.value).strip()


def inspect_with_xlrd(path: str) -> list[str]:
    import xlrd as _xlrd
    lines = []
    wb = _xlrd.open_workbook(path, formatting_info=True)

    lines.append(f"引擎      : xlrd {_xlrd.__version__}")
    lines.append(f"工作簿    : {os.path.basename(path)}")
    lines.append(f"Sheet 数  : {wb.nsheets}")
    lines.append(SEP_MAJOR)

    for si in range(wb.nsheets):
        ws = wb.sheet_by_index(si)
        lines.append(f"\n【Sheet {si}】名称: 《{ws.name}》  "
                     f"({ws.nrows} 行 × {ws.ncols} 列)")
        lines.append(SEP_MINOR)

        # 列宽参考（字符数）
        col_widths = []
        for ci in range(ws.ncols):
            max_w = 4
            for ri in range(ws.nrows):
                w = len(cell_str(ws.cell(ri, ci)))
                if w > max_w:
                    max_w = w
            col_widths.append(min(max_w + 2, 40))

        # 打印每行
        for ri in range(ws.nrows):
            row_parts = []
            for ci in range(ws.ncols):
                txt = cell_str(ws.cell(ri, ci))
                row_parts.append(txt.ljust(col_widths[ci]))
            row_line = f"  行{ri:03d} | " + " | ".join(row_parts)
            lines.append(row_line)

        # 合并单元格信息
        merged = ws.merged_cells  # list of (rlo,rhi,clo,chi)
        if merged:
            lines.append(f"\n  ── 合并单元格 ({len(merged)} 处) ──")
            for rlo, rhi, clo, chi in merged:
                val = cell_str(ws.cell(rlo, clo))
                lines.append(
                    f"    行[{rlo}:{rhi}) × 列[{clo}:{chi})  值=「{val}」"
                )

    return lines


def inspect_with_pandas(path: str) -> list[str]:
    lines = []
    lines.append(f"引擎      : pandas {pd.__version__} (xlrd 未安装，降级使用)")
    lines.append(f"工作簿    : {os.path.basename(path)}")
    lines.append(SEP_MAJOR)

    xl = pd.ExcelFile(path, engine='xlrd')
    lines.append(f"Sheet 数  : {len(xl.sheet_names)}")

    for name in xl.sheet_names:
        df = xl.parse(name, header=None, dtype=str).fillna('')
        lines.append(f"\n【Sheet】名称: 《{name}》  "
                     f"({len(df)} 行 × {len(df.columns)} 列)")
        lines.append(SEP_MINOR)
        lines.append(df.to_string(index=True, header=False))

    return lines


def main():
    if not os.path.exists(XLS_PATH):
        print(f"❌ 文件不存在: {XLS_PATH}")
        sys.exit(1)

    print(f"📂 正在探查: {XLS_PATH}\n")

    if xlrd:
        lines = inspect_with_xlrd(XLS_PATH)
    elif _HAS_PANDAS:
        lines = inspect_with_pandas(XLS_PATH)
    else:
        print("❌ 请先安装依赖: pip install xlrd pandas")
        sys.exit(1)

    output = "\n".join(lines)

    # 终端输出
    print(output)

    # 保存文件
    os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        f.write(output)

    print(f"\n\n✅ 已保存至: {OUTPUT_TXT}")


if __name__ == '__main__':
    main()
