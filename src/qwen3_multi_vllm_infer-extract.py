import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import os
import re
import json
from tqdm import tqdm
from dataclasses import asdict
from transformers import AutoProcessor
import argparse
import torch

# ──────────────────────────────────────────────
# Prompt（System 存放所有指令，User 注入 .md 内容）
# ──────────────────────────────────────────────
SYSTEM_PROMPT = r'''You are an expert Document Information Extraction AI. Your task is to accurately extract structured information from OCR-generated Markdown text of student certificates/papers. The output must be strictly in JSON format.

<preprocessing_rules>
The input OCR text may contain SEVERE reading order errors and fragmentation due to complex layouts. Before extracting, you MUST:
1. Fix Reverse Ordering: Text blocks might appear upside-down or in completely wrong order (e.g., the end of a competition name appears before the beginning). You must logically reconnect them.
2. Fix Line Breaks: Words or names might be split across lines. You must merge them into a complete word.
3. Clean English Names: For academic papers, author names may contain superscripts (e.g., $ ^{ID} $, *), affiliations, or titles (e.g., Member, IEEE). You must strip these noises and extract pure names.
</preprocessing_rules>

<extraction_rules>
Extract the following fields. If a field is not mentioned, output `null`.

1. "certificate_type": Classify into EXACTLY ONE of: "Competition" (竞赛), "Software Copyright" (软件著作权), "Patent" (专利), "Paper" (论文), "Honorary Title" (荣誉称号).
2. "name": The full, reconstructed name of the certificate. 
   - For Competitions: You MUST extract the MOST COMPLETE name. It MUST include the base name, the stage/level (e.g., 全国总决赛, 省赛), AND the specific track/category (e.g., xx赛项, 组别). Do not truncate it. (e.g., "XXXX大赛全国总决赛（XXXX赛项）").
   - For others: The full software name, patent name, paper title, or honorary title.
3. "students": List of student/author names. 
   - For Chinese text: Look for "作者", "发明人", "同学". 
   - For English Papers: Authors are usually listed directly below the title, separated by commas or "and". Extract ALL author names here as pure text.
   - STRICTLY maintain the original order.
4. "advisors": List of advisor/teacher names (e.g., 指导老师). STRICTLY maintain the original order. If none, output [].
5. "team_name": Participating team name. Output `null` if not applicable.
6. "issue_date": Date issued/published/accepted. Format as "YYYY-MM-DD", "YYYY-MM" or "YYYY".
7. "issuing_authority": List of issuing organizations (e.g., 颁发部门).
8. "award_level": Award grade (e.g., "一等奖", "二等奖"). For non-competitions without explicit grades, output `null`.
</extraction_rules>

<examples>
Example 1:
Input Text:
# 获奖证书
- CERTIFICATE OF HONOR—
乔硕
# 二等奖
特发此证，以资鼓励。
在“泰豪杯”北京科技大学第十一届单
片机应用大赛中荣获

Output JSON:
{
  "certificate_type": "Competition",
  "name": "“泰豪杯”北京科技大学第十一届单片机应用大赛",
  "students": ["乔硕"],
  "advisors": [],
  "team_name": null,
  "issue_date": null,
  "issuing_authority": [],
  "award_level": "二等奖"
}

Example 2:
Input Text:
2023年12月
北京市教育委员会
指导老师：徐立业 张希琛
网联车赛项）中成绩优异，特授予大赛一等奖。
在2023年北京市大学生工程实践与创新能力大赛（智能
袁启轩 陈亚龙 南珏 同学

Output JSON:
{
  "certificate_type": "Competition",
  "name": "2023年北京市大学生工程实践与创新能力大赛（智能网联车赛项）",
  "students": ["袁启轩", "陈亚龙", "南珏"],
  "advisors": ["徐立业", "张希琛"],
  "team_name": null,
  "issue_date": "2023-12",
  "issuing_authority": ["北京市教育委员会"],
  "award_level": "一等奖"
}

Example 3:
Input Text:
# Knowledge Embedding With Graph Convolutional Network and Bidirectional Gated Recurrent Unit for Fault Diagnosis
Jie Dong $ ^{ID} $ , Cuiping Chen $ ^{ID} $ , Chi Zhang $ ^{ID} $ , Jinyao Ma, and Kaixiang Peng $ ^{ID} $ , Member, IEEE
Abstract—The stability and reliability of modern industrial processes are key factors...

Output JSON:
{
  "certificate_type": "Paper",
  "name": "Knowledge Embedding With Graph Convolutional Network and Bidirectional Gated Recurrent Unit for Fault Diagnosis",
  "students": ["Jie Dong", "Cuiping Chen", "Chi Zhang", "Jinyao Ma", "Kaixiang Peng"],
  "advisors": [],
  "team_name": null,
  "issue_date": null,
  "issuing_authority": [],
  "award_level": null
}

Example 4:
Input Text:
2024年“西门子杯”中国智能制造挑战赛
北京科技大学 陈亚龙
参加2024年第十八届CIMC“西门子杯”中国智能制造挑战赛全国总决赛，荣获
智能制造工程设计与应用类赛项：信息化网络化方向（本科组）
二等奖
中国智能制造挑战全国总决赛组委会

Output JSON:
{
  "certificate_type": "Competition",
  "name": "2024年第十八届CIMC“西门子杯”中国智能制造挑战赛全国总决赛（智能制造工程设计与应用类赛项：信息化网络化方向（本科组））",
  "students": ["陈亚龙"],
  "advisors": [],
  "team_name": null,
  "issue_date": "2024",
  "issuing_authority": ["中国智能制造挑战全国总决赛组委会"],
  "award_level": "二等奖"
}
</examples>

<output_format>
Output ONLY a valid JSON object. Do not include markdown code blocks (e.g., ```json), explanations, or conversational text.
</output_format>
'''



# ──────────────────────────────────────────────
# 默认路径
# ──────────────────────────────────────────────
DEFAULT_INPUT_DIR  = "/mnt/tidal-alsh01/dataset/OCRData/page_label_project/AutoCert/results/firered_ocr/red_2b_test"
DEFAULT_OUTPUT_DIR = "/mnt/tidal-alsh01/dataset/OCRData/page_label_project/AutoCert/results/qwen_extract"


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-VL-4B 证书信息结构化提取")
    parser.add_argument("--model_dir",     type=str, required=True,            help="Qwen3-VL-4B-Instruct 模型路径")
    parser.add_argument("--processor_dir", type=str, required=True,            help="Processor 路径（通常与 model_dir 相同）")
    parser.add_argument("--input_dir",     type=str, default=DEFAULT_INPUT_DIR, help=".md 文件所在目录")
    parser.add_argument("--output_dir",    type=str, default=DEFAULT_OUTPUT_DIR,help=".json 输出目录")
    parser.add_argument("--overwrite",     action="store_true",                 help="重新处理已存在的 .json 文件")
    return parser.parse_args()


def collect_md_files(input_dir: str) -> list[str]:
    """递归收集所有 .md 文件"""
    md_files = []
    for root, _, files in os.walk(input_dir):
        for name in sorted(files):
            if name.lower().endswith(".md"):
                md_files.append(os.path.join(root, name))
    return sorted(md_files)


def split_list(data: list, n: int) -> list[list]:
    """均匀切分到 n 份"""
    return [data[i::n] for i in range(n)]


def extract_json(raw_text: str) -> dict:
    """
    从模型输出中解析 JSON。
    处理以下情况：
      1. Qwen3 thinking 标签残留：<think>...</think>
      2. Markdown 代码块包裹：```json ... ```
      3. 输出中混有多余文字
    """
    # 1. 去掉 thinking 块（若 enable_thinking 未完全关闭时的保险）
    text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

    # 2. 去掉 Markdown 代码块标记
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$",          "", text, flags=re.MULTILINE)
    text = text.strip()

    # 3. 直接尝试解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 4. 正则提取最外层 { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # 5. 实在解析失败，保存原始文本供排查
    return {
        "_parse_error": True,
        "_raw_output":  raw_text,
    }


# ──────────────────────────────────────────────
# 多进程 Worker
# ──────────────────────────────────────────────
def worker(
    rank:         int,
    gpu_id:       int,
    md_paths:     list[str],
    model_dir:    str,
    processor_dir:str,
    input_dir:    str,
    output_dir:   str,
    overwrite:    bool,
):
    # ⚠️ 必须在 import vllm / torch 之前设置，spawn 模式下每个子进程独立
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from vllm import LLM, EngineArgs, SamplingParams

    print(f"[Worker {rank}] GPU={gpu_id}  files={len(md_paths)}", flush=True)

    # ---------- 加载 Processor ----------
    processor = AutoProcessor.from_pretrained(processor_dir)

    # ---------- 初始化 vllm 引擎（纯文本，不需要 mm 配置）----------
    engine_args = EngineArgs(
        model=model_dir,
        max_model_len=32768,
        max_num_seqs=8,          # 纯文本吞吐比图像高，可适当调大
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
    )
    llm = LLM(**{**asdict(engine_args), "seed": 42})

    sampling_params = SamplingParams(
        temperature=0.0,         # greedy，保证 JSON 输出稳定
        max_tokens=2048,
        stop=["<|im_end|>"],     # Qwen 系列的结束符
    )

    # ---------- 逐文件处理 ----------
    for md_path in tqdm(md_paths, desc=f"Worker{rank}/GPU{gpu_id}"):

        # 计算输出路径（保留目录层级）
        rel_path      = os.path.relpath(md_path, input_dir)
        rel_no_ext    = os.path.splitext(rel_path)[0]
        json_out_path = os.path.join(output_dir, rel_no_ext + ".json")

        # 断点续跑：跳过已存在的文件
        if not overwrite and os.path.exists(json_out_path):
            continue

        os.makedirs(os.path.dirname(json_out_path), exist_ok=True)

        # 读取 .md 内容
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read().strip()
        except Exception as e:
            print(f"[Worker {rank}] ❌ 读取失败 {md_path}: {e}", flush=True)
            continue

        if not md_content:
            print(f"[Worker {rank}] ⚠️  空文件，跳过 {md_path}", flush=True)
            continue

        # 构建消息
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                # .md 文本直接作为 user 输入
                "content": f"Input Text:\n{md_content}",
            },
        ]

        # apply_chat_template
        # enable_thinking=False：关闭 Qwen3 的 CoT thinking 模式，直接输出 JSON
        try:
            prompt_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,   # ← Qwen3 专属参数，关闭 <think> 块
            )
        except TypeError:
            # 如果当前 tokenizer 版本不支持 enable_thinking，降级处理
            prompt_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # 推理（纯文本，无 multi_modal_data）
        try:
            outputs    = llm.generate(prompt_text, sampling_params=sampling_params)
            raw_output = outputs[0].outputs[0].text
        except Exception as e:
            print(f"[Worker {rank}] ❌ 推理失败 {md_path}: {e}", flush=True)
            raw_output = ""

        # 解析 JSON
        result = extract_json(raw_output)

        # 写出 .json
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[Worker {rank}] ✅ 完成", flush=True)


# ──────────────────────────────────────────────
# 主进程
# ──────────────────────────────────────────────
def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    md_paths = collect_md_files(args.input_dir)
    if not md_paths:
        raise FileNotFoundError(f"在 {args.input_dir} 下未找到任何 .md 文件")

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("未检测到 CUDA 设备")

    print(f"检测到 {num_gpus} 张 GPU，共 {len(md_paths)} 个 .md 文件", flush=True)

    chunks = split_list(md_paths, num_gpus)

    processes = []
    for rank, (gpu_id, chunk) in enumerate(zip(range(num_gpus), chunks)):
        if not chunk:
            continue
        p = mp.Process(
            target=worker,
            args=(
                rank,
                gpu_id,
                chunk,
                args.model_dir,
                args.processor_dir,
                args.input_dir,
                args.output_dir,
                args.overwrite,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("🎉 全部处理完成", flush=True)


if __name__ == "__main__":
    main()
