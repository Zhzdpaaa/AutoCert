#!/bin/bash

# ============================================
# PDF to Markdown 推理脚本
# ============================================

# PROCESSOR_DIR 的基础路径
PROCESSOR_BASE_DIR="/mnt/tidal-alsh01/dataset/OCRData/public_models"

# 日志目录
LOG_BASE_DIR="/mnt/tidal-alsh01/dataset/OCRData/page_label_project/AutoCert/logs/firered_ocr"

# 默认输入图片目录
DEFAULT_INPUT_DIR="/mnt/tidal-alsh01/dataset/OCRData/page_label_project/AutoCert/test_images"

# 帮助函数
show_help() {
    echo "Usage: $0 -v VERSION -m MODEL_DIR -p PROCESSOR_NAME [-i INPUT_DIR]"
    echo ""
    echo "Required Options:"
    echo "  -v, --version        模型版本 (例如: v117_6)"
    echo "  -m, --model_dir      模型地址 (完整路径)"
    echo "  -p, --processor_name 处理器模型名称 (例如: Qwen3-VL-4B-Instruct)"
    echo ""
    echo "Optional:"
    echo "  -i, --input_dir      输入图片目录 (默认: ${DEFAULT_INPUT_DIR})"
    echo "  -h, --help           显示帮助信息"
    echo ""
    echo "Example:"
    echo "  $0 -v v117_6 -m /path/to/checkpoint -p Qwen3-VL-4B-Instruct"
    echo "  $0 -v v117_6 -m /path/to/checkpoint -p Qwen3-VL-4B-Instruct -i /path/to/images"
    echo "  $0 --version v118_0 --model_dir /path/to/model --processor_name Qwen3-VL-2B-Instruct --input_dir /custom/input"
}

# 初始化变量为空（可选参数设置默认值）
VERSION=""
MODEL_DIR=""
PROCESSOR_NAME=""
INPUT_DIR="${DEFAULT_INPUT_DIR}"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -m|--model_dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        -p|--processor_name)
            PROCESSOR_NAME="$2"
            shift 2
            ;;
        -i|--input_dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查必须参数
MISSING_PARAMS=()

if [[ -z "${VERSION}" ]]; then
    MISSING_PARAMS+=("VERSION (-v)")
fi

if [[ -z "${MODEL_DIR}" ]]; then
    MISSING_PARAMS+=("MODEL_DIR (-m)")
fi

if [[ -z "${PROCESSOR_NAME}" ]]; then
    MISSING_PARAMS+=("PROCESSOR_NAME (-p)")
fi

if [[ ${#MISSING_PARAMS[@]} -gt 0 ]]; then
    echo "Error: Missing required parameters:"
    for param in "${MISSING_PARAMS[@]}"; do
        echo "  - ${param}"
    done
    echo ""
    show_help
    exit 1
fi

# 拼接完整的 PROCESSOR_DIR
PROCESSOR_DIR="${PROCESSOR_BASE_DIR}/${PROCESSOR_NAME}"

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 创建日志目录（如果不存在）
mkdir -p "${LOG_BASE_DIR}"

# 生成日志文件路径
LOG_FILE="${LOG_BASE_DIR}/${VERSION}_infer_${TIMESTAMP}.log"

# 输出结果目录
OUTPUT_DIR="/mnt/tidal-alsh01/dataset/OCRData/page_label_project/AutoCert/results/firered_ocr/${VERSION}"

# 创建输出目录（如果不存在）
mkdir -p "${OUTPUT_DIR}"

# 打印当前配置
echo "============================================"
echo "Current Configuration:"
echo "  VERSION:       ${VERSION}"
echo "  MODEL_DIR:     ${MODEL_DIR}"
echo "  PROCESSOR_DIR: ${PROCESSOR_DIR}"
echo "  INPUT_DIR:     ${INPUT_DIR}"
echo "  OUTPUT_DIR:    ${OUTPUT_DIR}"
echo "  LOG_FILE:      ${LOG_FILE}"
echo "============================================"

# Python脚本地址
PYTHON_SCRIPT="/mnt/tidal-alsh01/dataset/OCRData/page_label_project/AutoCert/qwen3_multi_vllm_infer-ocr.py"

# ============================================
# 运行推理
# ============================================

# 写入日志文件头部信息
{
    echo "============================================"
    echo "Inference Log"
    echo "============================================"
    echo "Start Time:    $(date '+%Y-%m-%d %H:%M:%S')"
    echo "VERSION:       ${VERSION}"
    echo "MODEL_DIR:     ${MODEL_DIR}"
    echo "PROCESSOR_DIR: ${PROCESSOR_DIR}"
    echo "INPUT_DIR:     ${INPUT_DIR}"
    echo "OUTPUT_DIR:    ${OUTPUT_DIR}"
    echo "============================================"
    echo ""
} > "${LOG_FILE}"

# 运行 Python 脚本，将 stdout 和 stderr 都重定向到日志文件
python "${PYTHON_SCRIPT}" \
    --model_dir "${MODEL_DIR}" \
    --processor_dir "${PROCESSOR_DIR}" \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    >> "${LOG_FILE}" 2>&1

# 保存退出码
EXIT_CODE=$?

# 写入日志文件尾部信息
{
    echo ""
    echo "============================================"
    echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Exit Code: ${EXIT_CODE}"
    echo "============================================"
} >> "${LOG_FILE}"

# 检查运行结果
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "============================================"
    echo "Inference completed successfully!"
    echo "Results saved to: ${OUTPUT_DIR}"
    echo "Log saved to: ${LOG_FILE}"
    echo "============================================"
else
    echo "============================================"
    echo "Error: Inference failed!"
    echo "Check log file: ${LOG_FILE}"
    echo "============================================"
    exit 1
fi
