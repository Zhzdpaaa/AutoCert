#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoCert Web Server
Flask 后端：图片上传 → 调用 pipeline.sh → JSON 展示 → 加分表比对
"""

from flask import Flask, request, jsonify, send_from_directory, Response
import os
import subprocess
import json
import uuid
import threading
import time
from pathlib import Path
from datetime import datetime

# ============================================================
# 配置
# ============================================================
BASE_DIR        = "/mnt/tidal-alsh01/dataset/OCRData/page_label_project/AutoCert"
PUBLIC_MODELS   = "/mnt/tidal-alsh01/dataset/OCRData/public_models"
UPLOAD_DIR      = os.path.join(BASE_DIR, "web_uploads")
RESULT_OCR_BASE = os.path.join(BASE_DIR, "results/firered_ocr")
RESULT_EXT_BASE = os.path.join(BASE_DIR, "results/qwen_extract")
PIPELINE_SCRIPT = os.path.join(BASE_DIR, "pipeline.sh")

DEFAULT_OCR_MODEL  = "/mnt/tidal-alsh01/dataset/OCRData/FireRed-OCR-2B"
DEFAULT_PROCESSOR  = "Qwen3-VL-2B-Instruct"

ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp', 'pdf'}

# ============================================================
# 加分标准表（可按需修改）
# ============================================================
SCORING_TABLE = {
    "职业技能等级证书": {
        "五级/初级工":  2,
        "四级/中级工":  4,
        "三级/高级工":  6,
        "二级/技师":    8,
        "一级/高级技师": 10,
    },
    "职业资格证书": {
        "初级": 3,
        "中级": 5,
        "高级": 7,
    },
    "学历证书": {
        "大专": 3,
        "本科": 5,
        "硕士研究生": 8,
        "博士研究生": 12,
    },
    "专业技术资格证书": {
        "助理级": 3,
        "中级":   6,
        "高级":   9,
        "正高级": 12,
    },
    "技能竞赛获奖证书": {
        "市级三等奖":  2,
        "市级二等奖":  3,
        "市级一等奖":  4,
        "省级三等奖":  5,
        "省级二等奖":  7,
        "省级一等奖":  9,
        "国家级三等奖": 10,
        "国家级二等奖": 12,
        "国家级一等奖": 15,
    },
}

# ============================================================
# Flask 初始化
# ============================================================
app = Flask(__name__, static_folder="static")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

# 内存中保存 job 状态
pipeline_jobs: dict = {}


# ============================================================
# 工具函数
# ============================================================
def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def strip_ansi(text: str) -> str:
    """去掉终端彩色转义码"""
    import re
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


# ============================================================
# 路由：静态页面
# ============================================================
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ============================================================
# 路由：上传图片
# ============================================================
@app.route("/api/upload", methods=["POST"])
def upload_images():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "未收到文件"}), 400

    session_id  = request.form.get("session_id") or str(uuid.uuid4())
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    saved = []
    skipped = []
    for f in files:
        if f and allowed(f.filename):
            dest = os.path.join(session_dir, f.filename)
            f.save(dest)
            saved.append(f.filename)
        else:
            skipped.append(f.filename)

    return jsonify({
        "session_id": session_id,
        "input_dir":  session_dir,
        "saved":      saved,
        "skipped":    skipped,
    })


# ============================================================
# 路由：启动 Pipeline
# ============================================================
@app.route("/api/run", methods=["POST"])
def run_pipeline():
    body = request.get_json(force=True)

    input_dir     = body.get("input_dir", "")
    version       = body.get("version") or f"web_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    skip_ocr      = body.get("skip_ocr", False)
    only_ocr      = body.get("only_ocr", False)
    ocr_model_dir = body.get("ocr_model_dir", "")
    processor     = body.get("processor_name", "")

    if not input_dir or not os.path.isdir(input_dir):
        return jsonify({"error": f"输入目录不存在: {input_dir}"}), 400

    # 构造命令
    cmd = ["bash", PIPELINE_SCRIPT, "-v", version, "-i", input_dir]
    if ocr_model_dir: cmd += ["-m", ocr_model_dir]
    if processor:     cmd += ["-p", processor]
    if skip_ocr:      cmd.append("--skip_ocr")
    if only_ocr:      cmd.append("--only_ocr")

    job_id = str(uuid.uuid4())
    pipeline_jobs[job_id] = {
        "status":     "running",
        "version":    version,
        "log":        [],
        "start_time": datetime.now().isoformat(),
        "cmd":        " ".join(cmd),
    }

    def _run():
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            pipeline_jobs[job_id]["pid"] = proc.pid
            for line in proc.stdout:
                pipeline_jobs[job_id]["log"].append(strip_ansi(line.rstrip()))
            proc.wait()
            code = proc.returncode
            pipeline_jobs[job_id].update({
                "status":   "success" if code == 0 else "failed",
                "exit_code": code,
                "end_time":  datetime.now().isoformat(),
            })
        except Exception as exc:
            pipeline_jobs[job_id].update({"status": "failed", "error": str(exc)})

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return jsonify({"job_id": job_id, "version": version})


# ============================================================
# 路由：查询 Job 状态
# ============================================================
@app.route("/api/job/<job_id>", methods=["GET"])
def job_status(job_id):
    job = pipeline_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job 不存在"}), 404
    return jsonify(job)


# ============================================================
# 路由：SSE 实时日志流
# ============================================================
@app.route("/api/job/<job_id>/stream")
def stream_log(job_id):
    def _generate():
        cursor = 0
        while True:
            job = pipeline_jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'error': 'Job 不存在'})}\n\n"
                return

            logs = job["log"]
            while cursor < len(logs):
                yield f"data: {json.dumps({'line': logs[cursor]})}\n\n"
                cursor += 1  # noqa: F821 — nonlocal not needed, closure works

            if job["status"] in ("success", "failed"):
                yield f"data: {json.dumps({'done': True, 'status': job['status'], 'exit_code': job.get('exit_code', -1)})}\n\n"
                return

            time.sleep(0.4)

    # SSE 需要用 nonlocal，这里用列表包装 cursor
    def _generate_fixed():
        state = {"cursor": 0}
        while True:
            job = pipeline_jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'error': 'Job 不存在'})}\n\n"
                return

            logs = job["log"]
            while state["cursor"] < len(logs):
                yield f"data: {json.dumps({'line': logs[state['cursor']]})}\n\n"
                state["cursor"] += 1

            if job["status"] in ("success", "failed"):
                yield f"data: {json.dumps({'done': True, 'status': job['status'], 'exit_code': job.get('exit_code', -1)})}\n\n"
                return

            time.sleep(0.4)

    return Response(
        _generate_fixed(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============================================================
# 路由：获取提取结果 JSON
# ============================================================
@app.route("/api/results/<version>", methods=["GET"])
def get_results(version):
    result_dir = Path(RESULT_EXT_BASE) / version
    if not result_dir.exists():
        return jsonify({"error": f"结果目录不存在: {result_dir}"}), 404

    json_files = sorted(result_dir.glob("*.json"))
    if not json_files:
        return jsonify({"error": "目录中无 JSON 文件"}), 404

    data = {}
    for jf in json_files:
        try:
            data[jf.stem] = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            data[jf.stem] = {"_error": str(e)}

    return jsonify(data)


# ============================================================
# 路由：加分标准表
# ============================================================
@app.route("/api/scoring-table", methods=["GET"])
def scoring_table():
    return jsonify(SCORING_TABLE)


# ============================================================
# 路由：比对加分表
# ============================================================
@app.route("/api/compare", methods=["POST"])
def compare():
    body          = request.get_json(force=True)
    extracted     = body.get("extracted_data", {})    # { filename: {...} }
    scoring_rules = body.get("scoring_table", SCORING_TABLE)

    results     = []
    total_score = 0

    for filename, cert in extracted.items():
        if not isinstance(cert, dict) or "_error" in cert:
            results.append({"filename": filename, "matched": False,
                            "score": 0, "reason": "JSON 解析异常"})
            continue

        cert_type = cert.get("证书类型", cert.get("certificate_type", "未知"))
        level     = cert.get("级别",    cert.get("level",            "未知"))
        name      = cert.get("姓名",    cert.get("name",             "—"))

        score   = 0
        matched = False
        hint    = "未匹配到加分规则"

        for rule_type, rule_levels in scoring_rules.items():
            if rule_type in cert_type or cert_type in rule_type:
                for rule_level, pts in rule_levels.items():
                    if rule_level in level or level in rule_level:
                        score   = pts
                        matched = True
                        hint    = f"{rule_type} · {rule_level} → +{pts} 分"
                        break
            if matched:
                break

        total_score += score
        results.append({
            "filename":  filename,
            "name":      name,
            "cert_type": cert_type,
            "level":     level,
            "matched":   matched,
            "score":     score,
            "hint":      hint,
            "raw":       cert,
        })

    return jsonify({"results": results, "total_score": total_score})


# ============================================================
# 路由：默认配置（供前端初始化）
# ============================================================
@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify({
        "default_ocr_model": DEFAULT_OCR_MODEL,
        "default_processor": DEFAULT_PROCESSOR,
    })


# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
