#!/bin/bash

# ============================================================
# AutoCert Pipeline: OCR → 信息提取
# Stage 1: FireRed-OCR_2B  (图片 → .md)
# Stage 2: Qwen3-VL Extract (  .md → .json)
# ============================================================

# ============================================================
# ① 全局固定路径配置
# ============================================================
BASE_DIR="/mnt/tidal-alsh01/dataset/OCRData/page_label_project/AutoCert"
PUBLIC_MODELS_DIR="/mnt/tidal-alsh01/dataset/OCRData/public_models"

# Python 脚本
OCR_PYTHON_SCRIPT="/mnt/tidal-alsh01/dataset/OCRData/page_label_project/AutoCert/src/qwen3_multi_vllm_infer-ocr.py"
EXTRACT_PYTHON_SCRIPT="/mnt/tidal-alsh01/dataset/OCRData/page_label_project/AutoCert/src/qwen3_multi_vllm_infer-extract.py"

# 日志根目录
LOG_OCR_DIR="${BASE_DIR}/logs/firered_ocr"
LOG_EXTRACT_DIR="${BASE_DIR}/logs/qwen_extract"

# 结果根目录
RESULT_OCR_BASE="${BASE_DIR}/results/firered_ocr"
RESULT_EXTRACT_BASE="${BASE_DIR}/results/qwen_extract"

# Stage 1 OCR 默认模型路径 / 默认 Processor 名称
DEFAULT_OCR_MODEL_DIR="/mnt/tidal-alsh01/dataset/OCRData/FireRed-OCR-2B" 
DEFAULT_PROCESSOR_NAME="Qwen3-VL-2B-Instruct"

# Stage 2 固定模型（语言提取阶段始终用 Qwen3-VL-4B-Instruct）
EXTRACT_MODEL_DIR="${PUBLIC_MODELS_DIR}/Qwen3-VL-4B-Instruct"
EXTRACT_PROCESSOR_DIR="${PUBLIC_MODELS_DIR}/Qwen3-VL-4B-Instruct"

# ============================================================
# ② 帮助信息
# ============================================================
show_help() {
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "必填参数:"
    echo "  -v, --version           OCR 模型版本标识  (例: v117_6)"
    echo "  -i, --input_dir         输入图片目录       (必填，无默认值)"
    echo ""
    echo "可选参数 (Stage 1 - OCR):"
    echo "  -m, --ocr_model_dir     OCR 模型完整路径  (默认: ${DEFAULT_OCR_MODEL_DIR})"
    echo "  -p, --processor_name    处理器名称        (默认: ${DEFAULT_PROCESSOR_NAME})"
    echo ""
    echo "其他可选参数:"
    echo "  -e, --extract_version   提取结果版本标识   (默认: 与 --version 相同)"
    echo "  -s, --skip_ocr          跳过 OCR 阶段，直接运行提取阶段"
    echo "  -o, --only_ocr          仅运行 OCR 阶段，不运行提取阶段"
    echo "  -h, --help              显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # 完整流水线（仅必填参数）"
    echo "  $0 -v v117_6 -i /path/to/images"
    echo ""
    echo "  # 指定自定义模型 + 自定义提取版本"
    echo "  $0 -v v117_6 -i /path/to/images \\"
    echo "     -m /path/to/checkpoint -p Qwen3-VL-4B-Instruct -e extract_v1"
    echo ""
    echo "  # 跳过 OCR，直接用已有 OCR 结果做提取"
    echo "  $0 -v v117_6 -i /path/to/images --skip_ocr"
    echo ""
    echo "  # 仅做 OCR，不做提取"
    echo "  $0 -v v117_6 -i /path/to/images --only_ocr"
    echo ""
}

# ============================================================
# ③ 彩色日志工具函数
# ============================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

log_info()    { echo -e "${BLUE}[INFO ]${NC} $(date '+%H:%M:%S') $*"; }
log_success() { echo -e "${GREEN}[OK   ]${NC} $(date '+%H:%M:%S') $*"; }
log_warn()    { echo -e "${YELLOW}[WARN ]${NC} $(date '+%H:%M:%S') $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $*"; }
log_stage()   { echo -e "\n${BOLD}${CYAN}$*${NC}\n"; }
log_sep()     { echo -e "${CYAN}============================================================${NC}"; }

# ============================================================
# ④ 解析命令行参数
# ============================================================
VERSION=""
OCR_MODEL_DIR="${DEFAULT_OCR_MODEL_DIR}"      # 非必填，有默认值
PROCESSOR_NAME="${DEFAULT_PROCESSOR_NAME}"     # 非必填，有默认值
INPUT_DIR=""                                   # 必填，无默认值
EXTRACT_VERSION=""
SKIP_OCR=false
ONLY_OCR=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--version)
            VERSION="$2"; shift 2 ;;
        -m|--ocr_model_dir)
            OCR_MODEL_DIR="$2"; shift 2 ;;
        -p|--processor_name)
            PROCESSOR_NAME="$2"; shift 2 ;;
        -i|--input_dir)
            INPUT_DIR="$2"; shift 2 ;;
        -e|--extract_version)
            EXTRACT_VERSION="$2"; shift 2 ;;
        -s|--skip_ocr)
            SKIP_OCR=true; shift ;;
        -o|--only_ocr)
            ONLY_OCR=true; shift ;;
        -h|--help)
            show_help; exit 0 ;;
        *)
            log_error "未知参数: $1"
            show_help; exit 1 ;;
    esac
done

# ============================================================
# ⑤ 参数校验
# ============================================================
MISSING_PARAMS=()

# -v 和 -i 为必填；-m 和 -p 已有默认值，无需校验
[[ -z "${VERSION}" ]]   && MISSING_PARAMS+=("VERSION    (-v)")
[[ -z "${INPUT_DIR}" ]] && MISSING_PARAMS+=("INPUT_DIR  (-i)")

if [[ ${#MISSING_PARAMS[@]} -gt 0 ]]; then
    log_error "缺少必填参数:"
    for p in "${MISSING_PARAMS[@]}"; do
        echo "    - ${p}"
    done
    show_help; exit 1
fi

if [[ "${SKIP_OCR}" == true && "${ONLY_OCR}" == true ]]; then
    log_error "--skip_ocr 与 --only_ocr 不能同时使用"
    exit 1
fi

# 提取阶段版本标识，默认与 OCR 版本相同
[[ -z "${EXTRACT_VERSION}" ]] && EXTRACT_VERSION="${VERSION}"

# ============================================================
# ⑥ 路径推导
# ============================================================
PROCESSOR_DIR="${PUBLIC_MODELS_DIR}/${PROCESSOR_NAME}"

# Stage 1 输出 = Stage 2 输入
OCR_OUTPUT_DIR="${RESULT_OCR_BASE}/${VERSION}"

# Stage 2 输出
EXTRACT_OUTPUT_DIR="${RESULT_EXTRACT_BASE}/${EXTRACT_VERSION}"

# 时间戳（整条流水线共享同一时间戳，便于追踪）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 日志文件
LOG_OCR_FILE="${LOG_OCR_DIR}/${VERSION}_infer_${TIMESTAMP}.log"
LOG_EXTRACT_FILE="${LOG_EXTRACT_DIR}/${EXTRACT_VERSION}_extract_${TIMESTAMP}.log"
LOG_PIPELINE_FILE="${BASE_DIR}/logs/pipeline_${VERSION}_${TIMESTAMP}.log"

# ============================================================
# ⑦ 创建必要目录
# ============================================================
mkdir -p "${LOG_OCR_DIR}"
mkdir -p "${LOG_EXTRACT_DIR}"
mkdir -p "${BASE_DIR}/logs"
mkdir -p "${OCR_OUTPUT_DIR}"
mkdir -p "${EXTRACT_OUTPUT_DIR}"

# ============================================================
# ⑧ 打印流水线配置总览
# ============================================================
log_sep
echo -e "${BOLD}          AutoCert 推理流水线配置总览${NC}"
log_sep
echo -e "  ${BOLD}[Pipeline]${NC}"
echo    "    时间戳          : ${TIMESTAMP}"
echo    "    流水线日志       : ${LOG_PIPELINE_FILE}"
echo ""
echo -e "  ${BOLD}[Stage 1 - OCR]${NC}"
echo    "    跳过            : ${SKIP_OCR}"
echo    "    版本            : ${VERSION}"
echo    "    OCR 模型        : ${OCR_MODEL_DIR}"
echo    "    Processor       : ${PROCESSOR_DIR}"
echo    "    输入图片目录     : ${INPUT_DIR}"
echo    "    输出 .md 目录   : ${OCR_OUTPUT_DIR}"
echo    "    日志            : ${LOG_OCR_FILE}"
echo ""
echo -e "  ${BOLD}[Stage 2 - 信息提取]${NC}"
echo    "    跳过            : ${ONLY_OCR}"
echo    "    版本            : ${EXTRACT_VERSION}"
echo    "    提取模型        : ${EXTRACT_MODEL_DIR}"
echo    "    输入 .md 目录   : ${OCR_OUTPUT_DIR}"
echo    "    输出 .json 目录 : ${EXTRACT_OUTPUT_DIR}"
echo    "    日志            : ${LOG_EXTRACT_FILE}"
log_sep

# ============================================================
# ⑨ 流水线执行函数
# ============================================================

# ---------- 写日志头 ----------
write_log_header() {
    local log_file="$1"
    local stage="$2"
    {
        echo "============================================================"
        echo "  ${stage} Log"
        echo "============================================================"
        echo "  Start Time : $(date '+%Y-%m-%d %H:%M:%S')"
        echo "  VERSION    : ${VERSION}"
        echo "  TIMESTAMP  : ${TIMESTAMP}"
        echo "============================================================"
        echo ""
    } > "${log_file}"
}

# ---------- 写日志尾 ----------
write_log_footer() {
    local log_file="$1"
    local exit_code="$2"
    {
        echo ""
        echo "============================================================"
        echo "  End Time   : $(date '+%Y-%m-%d %H:%M:%S')"
        echo "  Exit Code  : ${exit_code}"
        echo "============================================================"
    } >> "${log_file}"
}

# ============================================================
# ⑩ Stage 1: OCR 推理
# ============================================================
run_ocr_stage() {
    log_stage ">>> Stage 1 / 2 : FireRed-OCR_2B 推理 (图片 → Markdown)"
    log_info "FireRed-OCR_2B 模型  : ${OCR_MODEL_DIR}"
    log_info "输入图片目录          : ${INPUT_DIR}"
    log_info "输出 .md 目录         : ${OCR_OUTPUT_DIR}"
    log_info "日志文件              : ${LOG_OCR_FILE}"

    # 写日志头
    write_log_header "${LOG_OCR_FILE}" "Stage-1 OCR"

    # 附加 OCR 相关配置到日志
    {
        echo "  OCR_MODEL_DIR  : ${OCR_MODEL_DIR}"
        echo "  PROCESSOR_DIR  : ${PROCESSOR_DIR}"
        echo "  INPUT_DIR      : ${INPUT_DIR}"
        echo "  OUTPUT_DIR     : ${OCR_OUTPUT_DIR}"
        echo ""
    } >> "${LOG_OCR_FILE}"

    # 执行推理
    python "${OCR_PYTHON_SCRIPT}" \
        --model_dir     "${OCR_MODEL_DIR}" \
        --processor_dir "${PROCESSOR_DIR}" \
        --input_dir     "${INPUT_DIR}" \
        --output_dir    "${OCR_OUTPUT_DIR}" \
        >> "${LOG_OCR_FILE}" 2>&1

    local exit_code=$?
    write_log_footer "${LOG_OCR_FILE}" "${exit_code}"

    if [[ ${exit_code} -eq 0 ]]; then
        log_success "Stage 1 完成！.md 文件已保存至: ${OCR_OUTPUT_DIR}"
        # 统计输出文件数量
        local md_count
        md_count=$(find "${OCR_OUTPUT_DIR}" -maxdepth 1 -name "*.md" 2>/dev/null | wc -l)
        log_info "共生成 ${md_count} 个 .md 文件"
    else
        log_error "Stage 1 失败！退出码: ${exit_code}"
        log_error "请检查日志: ${LOG_OCR_FILE}"
        return ${exit_code}
    fi

    return 0
}

# ============================================================
# ⑪ Stage 2: 信息提取
# ============================================================
run_extract_stage() {
    log_stage ">>> Stage 2 / 2 : Qwen3 信息提取 (Markdown → JSON)"
    log_info "提取模型          : ${EXTRACT_MODEL_DIR}"
    log_info "输入 .md 目录    : ${OCR_OUTPUT_DIR}"
    log_info "输出 .json 目录  : ${EXTRACT_OUTPUT_DIR}"
    log_info "日志文件          : ${LOG_EXTRACT_FILE}"

    # 检查输入目录是否存在且非空
    if [[ ! -d "${OCR_OUTPUT_DIR}" ]]; then
        log_error "输入目录不存在: ${OCR_OUTPUT_DIR}"
        return 1
    fi

    local md_count
    md_count=$(find "${OCR_OUTPUT_DIR}" -maxdepth 1 -name "*.md" 2>/dev/null | wc -l)
    if [[ ${md_count} -eq 0 ]]; then
        log_warn "输入目录中没有找到 .md 文件: ${OCR_OUTPUT_DIR}"
        log_warn "请确认 Stage 1 已正确运行，或手动指定正确的 OCR 输出目录"
        return 1
    fi
    log_info "检测到 ${md_count} 个 .md 文件，开始提取..."

    # 写日志头
    write_log_header "${LOG_EXTRACT_FILE}" "Stage-2 Extract"

    {
        echo "  EXTRACT_MODEL  : ${EXTRACT_MODEL_DIR}"
        echo "  INPUT_DIR      : ${OCR_OUTPUT_DIR}"
        echo "  OUTPUT_DIR     : ${EXTRACT_OUTPUT_DIR}"
        echo ""
    } >> "${LOG_EXTRACT_FILE}"

    # 切换到工作目录
    cd "${BASE_DIR}" || { log_error "无法切换到: ${BASE_DIR}"; return 1; }

    # 执行信息提取
    python "${EXTRACT_PYTHON_SCRIPT}" \
        --model_dir     "${EXTRACT_MODEL_DIR}" \
        --processor_dir "${EXTRACT_PROCESSOR_DIR}" \
        --input_dir     "${OCR_OUTPUT_DIR}" \
        --output_dir    "${EXTRACT_OUTPUT_DIR}" \
        >> "${LOG_EXTRACT_FILE}" 2>&1

    local exit_code=$?
    write_log_footer "${LOG_EXTRACT_FILE}" "${exit_code}"

    if [[ ${exit_code} -eq 0 ]]; then
        log_success "Stage 2 完成！.json 文件已保存至: ${EXTRACT_OUTPUT_DIR}"
        local json_count
        json_count=$(find "${EXTRACT_OUTPUT_DIR}" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l)
        log_info "共生成 ${json_count} 个 .json 文件"
    else
        log_error "Stage 2 失败！退出码: ${exit_code}"
        log_error "请检查日志: ${LOG_EXTRACT_FILE}"
        return ${exit_code}
    fi

    return 0
}

# ============================================================
# ⑫ 流水线总日志初始化
# ============================================================
{
    echo "============================================================"
    echo "  AutoCert Pipeline Log"
    echo "============================================================"
    echo "  Start Time      : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  VERSION         : ${VERSION}"
    echo "  EXTRACT_VERSION : ${EXTRACT_VERSION}"
    echo "  TIMESTAMP       : ${TIMESTAMP}"
    echo "  SKIP_OCR        : ${SKIP_OCR}"
    echo "  ONLY_OCR        : ${ONLY_OCR}"
    echo "============================================================"
} > "${LOG_PIPELINE_FILE}"

# ============================================================
# ⑬ 执行流水线
# ============================================================
PIPELINE_START=$(date +%s)
OVERALL_EXIT=0

# ---------- Stage 1 ----------
if [[ "${SKIP_OCR}" == false ]]; then
    run_ocr_stage
    STAGE1_EXIT=$?
    echo "Stage-1 Exit: ${STAGE1_EXIT}" >> "${LOG_PIPELINE_FILE}"

    if [[ ${STAGE1_EXIT} -ne 0 ]]; then
        log_error "Stage 1 失败，流水线终止"
        echo "Pipeline terminated at Stage-1" >> "${LOG_PIPELINE_FILE}"
        OVERALL_EXIT=${STAGE1_EXIT}
        # 跳过 Stage 2
        ONLY_OCR=true
    fi
else
    log_warn "已跳过 Stage 1 (--skip_ocr)，使用已有 OCR 结果: ${OCR_OUTPUT_DIR}"
    echo "Stage-1 Skipped" >> "${LOG_PIPELINE_FILE}"
fi

# ---------- Stage 2 ----------
if [[ "${ONLY_OCR}" == false ]]; then
    run_extract_stage
    STAGE2_EXIT=$?
    echo "Stage-2 Exit: ${STAGE2_EXIT}" >> "${LOG_PIPELINE_FILE}"

    if [[ ${STAGE2_EXIT} -ne 0 ]]; then
        log_error "Stage 2 失败"
        OVERALL_EXIT=${STAGE2_EXIT}
    fi
else
    if [[ "${SKIP_OCR}" == false ]]; then
        log_warn "仅运行 OCR 阶段 (--only_ocr)，跳过信息提取"
    fi
    echo "Stage-2 Skipped" >> "${LOG_PIPELINE_FILE}"
fi

# ============================================================
# ⑭ 流水线总结
# ============================================================
PIPELINE_END=$(date +%s)
ELAPSED=$(( PIPELINE_END - PIPELINE_START ))
ELAPSED_FMT=$(printf "%02d:%02d:%02d" $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))

{
    echo ""
    echo "============================================================"
    echo "  End Time        : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Total Elapsed   : ${ELAPSED_FMT}"
    echo "  Overall Exit    : ${OVERALL_EXIT}"
    echo "============================================================"
} >> "${LOG_PIPELINE_FILE}"

log_sep
if [[ ${OVERALL_EXIT} -eq 0 ]]; then
    echo -e "${BOLD}${GREEN}"
    echo    "  ✅  流水线全部完成！"
    echo -e "${NC}"
    echo    "  OCR 结果  → ${OCR_OUTPUT_DIR}"
    echo    "  提取结果  → ${EXTRACT_OUTPUT_DIR}"
    echo    "  总耗时    : ${ELAPSED_FMT}"
    echo    "  流水线日志: ${LOG_PIPELINE_FILE}"
else
    echo -e "${BOLD}${RED}"
    echo    "  ❌  流水线执行失败！"
    echo -e "${NC}"
    echo    "  请检查以下日志排查问题:"
    echo    "    Stage-1 日志 : ${LOG_OCR_FILE}"
    echo    "    Stage-2 日志 : ${LOG_EXTRACT_FILE}"
    echo    "    流水线日志   : ${LOG_PIPELINE_FILE}"
fi
log_sep

exit ${OVERALL_EXIT}
