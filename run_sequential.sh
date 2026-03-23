#!/bin/bash

# ============================================================
# 顺序执行多个脚本的通用执行器
# 
# 使用方法:
#   ./run_sequential.sh script1.sh script2.sh script3.sh ...
#   ./run_sequential.sh -c config.txt
#   ./run_sequential.sh -o /path/to/my_log.log script1.sh script2.sh
#
# 配置文件格式 (config.txt):
#   # 这是注释
#   train.sh
#   run_eval.sh /path/to/data --option value
#   ./another_script.sh arg1 arg2
#
# 选项:
#   -c, --config FILE       从配置文件读取脚本列表
#   -s, --stop-on-error     某个脚本失败后,不再执行后续脚本
#   -l, --log-dir DIR       日志目录（默认: ./logs）
#   -o, --output FILE       指定完整的日志文件路径（包括文件名）
#   -h, --help              显示此帮助信息
# ============================================================

set -o pipefail

# 默认配置
CONTINUE_ON_ERROR=true
CONFIG_FILE=""
SCRIPTS=()
LOG_DIR="./logs"
LOG_FILE=""
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 显示帮助
show_help() {
    echo "用法: $0 [选项] script1.sh script2.sh ..."
    echo ""
    echo "选项:"
    echo "  -c, --config FILE       从配置文件读取脚本列表（每行一个脚本，可带参数）"
    echo "  -s, --stop-on-error     某个脚本失败后,不再执行后续脚本"
    echo "  -l, --log-dir DIR       日志目录（默认: ./logs，文件名自动生成）"
    echo "  -o, --output FILE       指定完整的日志文件路径（包括文件名）"
    echo "  -h, --help              显示此帮助信息"
    echo ""
    echo "配置文件格式:"
    echo "  # 注释行"
    echo "  train.sh"
    echo "  run_eval.sh /path/to/data --option value"
    echo ""
    echo "示例:"
    echo "  $0 train1.sh train2.sh train3.sh"
    echo "  $0 -c scripts.txt"
    echo "  $0 -l /path/to/logs script1.sh script2.sh"
    echo "  $0 -o /path/to/my_training.log script1.sh script2.sh"
    exit 0
}

# 日志函数（只写入文件，不输出到终端）
log() {
    local level=$1
    local msg=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        INFO)  echo "[${timestamp}] [INFO] $msg" >> "${LOG_FILE}" ;;
        OK)    echo "[${timestamp}] [✓ OK] $msg" >> "${LOG_FILE}" ;;
        WARN)  echo "[${timestamp}] [WARN] $msg" >> "${LOG_FILE}" ;;
        ERROR) echo "[${timestamp}] [✗ ERROR] $msg" >> "${LOG_FILE}" ;;
    esac
}

# 解析参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -s|--stop-on-error)
                CONTINUE_ON_ERROR=false
                shift
                ;;
            -l|--log-dir)
                LOG_DIR="$2"
                shift 2
                ;;
            -o|--output)
                LOG_FILE="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                ;;
            -*)
                echo "未知选项: $1"
                show_help
                ;;
            *)
                SCRIPTS+=("$1")
                shift
                ;;
        esac
    done
}

# 从配置文件读取脚本列表（支持带参数的命令）
load_config() {
    if [[ -n "$CONFIG_FILE" ]]; then
        if [[ ! -f "$CONFIG_FILE" ]]; then
            echo "配置文件不存在: $CONFIG_FILE"
            exit 1
        fi
        
        while IFS= read -r line || [[ -n "$line" ]]; do
            # 去除行首尾空白
            line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            
            # 跳过空行和注释行
            if [[ -z "$line" ]] || [[ "$line" == \#* ]]; then
                continue
            fi
            
            # 保留完整的命令行（包括参数）
            SCRIPTS+=("$line")
        done < "$CONFIG_FILE"
    fi
}

# 设置日志文件路径
setup_log_file() {
    if [[ -n "$LOG_FILE" ]]; then
        local log_dir=$(dirname "$LOG_FILE")
        mkdir -p "$log_dir"
    else
        mkdir -p "$LOG_DIR"
        LOG_FILE="${LOG_DIR}/sequential_${TIMESTAMP}.log"
    fi
}

# 执行单个脚本（支持带参数）
run_script() {
    local cmd=$1
    local index=$2
    local total=$3
    
    # 提取脚本路径（命令的第一个部分）
    local script=$(echo "$cmd" | awk '{print $1}')
    
    log INFO "=========================================="
    log INFO "[$index/$total] 开始执行: $cmd"
    log INFO "=========================================="
    
    # 检查脚本是否存在
    if [[ ! -f "$script" ]]; then
        log ERROR "脚本不存在: $script"
        return 1
    fi
    
    # 执行脚本（只输出到日志文件）
    local start_time=$(date +%s)
    
    # 使用 eval 执行完整命令（支持参数）
    bash -c "$cmd" >> "${LOG_FILE}" 2>&1
    local exit_code=$?
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_fmt=$(printf '%02d:%02d:%02d' $((duration/3600)) $((duration%3600/60)) $((duration%60)))
    
    if [[ $exit_code -eq 0 ]]; then
        log OK "[$index/$total] 完成: $cmd (耗时: $duration_fmt)"
    else
        log ERROR "[$index/$total] 失败: $cmd (退出码: $exit_code, 耗时: $duration_fmt)"
    fi
    
    return $exit_code
}

# 主函数
main() {
    parse_args "$@"
    load_config
    
    # 检查是否有脚本要执行
    if [[ ${#SCRIPTS[@]} -eq 0 ]]; then
        echo "错误: 没有指定要执行的脚本"
        show_help
    fi
    
    # 设置日志文件
    setup_log_file
    
    local total=${#SCRIPTS[@]}
    local success=0
    local failed=0
    local failed_scripts=()
    
    log INFO "=========================================="
    log INFO "顺序执行器启动"
    log INFO "总共 $total 个脚本待执行"
    log INFO "日志文件: $LOG_FILE"
    log INFO "失败后继续: $CONTINUE_ON_ERROR"
    log INFO "=========================================="
    
    # 列出所有脚本
    log INFO "待执行脚本列表:"
    for i in "${!SCRIPTS[@]}"; do
        log INFO "  $((i+1)). ${SCRIPTS[$i]}"
    done
    
    local start_total=$(date +%s)
    
    # 执行每个脚本
    for i in "${!SCRIPTS[@]}"; do
        local cmd="${SCRIPTS[$i]}"
        local index=$((i+1))
        
        run_script "$cmd" "$index" "$total"
        local exit_code=$?
        
        if [[ $exit_code -eq 0 ]]; then
            ((success++))
        else
            ((failed++))
            failed_scripts+=("$cmd")
            
            if [[ "$CONTINUE_ON_ERROR" == false ]]; then
                log ERROR "执行中止"
                break
            fi
        fi
    done
    
    local end_total=$(date +%s)
    local total_duration=$((end_total - start_total))
    local total_duration_fmt=$(printf '%02d:%02d:%02d' $((total_duration/3600)) $((total_duration%3600/60)) $((total_duration%60)))
    
    # 打印总结
    log INFO "=========================================="
    log INFO "执行总结"
    log INFO "=========================================="
    log INFO "总脚本数: $total"
    log OK   "成功: $success"
    
    if [[ $failed -gt 0 ]]; then
        log ERROR "失败: $failed"
        log ERROR "失败的脚本:"
        for script in "${failed_scripts[@]}"; do
            log ERROR "  - $script"
        done
    fi
    
    log INFO "总耗时: $total_duration_fmt"
    log INFO "日志文件: $LOG_FILE"
    log INFO "=========================================="
    
    if [[ $failed -gt 0 ]]; then
        exit 1
    fi
    exit 0
}

main "$@"
