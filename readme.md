# 🏆 AutoCert

<div align="center">

## **智能证书识别与保研加分计算系统**

*Automated Certificate Recognition and Graduate Recommendation Score Calculator*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/LLM-Qwen3--VL--4B-green.svg)](https://github.com/QwenLM/Qwen3)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/your-repo/AutoCert/pulls)

[Features](#-features) • [Quick Start](#-quick-start) • [Pipeline](#-pipeline) • [Output Format](#-output-format) • [Roadmap](#-roadmap)

</div>

---

## 📖 Overview

**AutoCert** 是一个端到端的智能证书识别与结构化信息提取系统，专为高校保研加分场景设计。系统通过两阶段 Pipeline 实现从证书图片到结构化 JSON 的全自动转换，最终对照加分规则表，自动计算每位成员的保研加分。

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  📷 Image   │ ──▶ │ 🔍 FireRed   │ ──▶ │ 🤖 Qwen3-VL-4B  │ ──▶ │ 📊 Score     │
│ Certificate │     │    OCR       │     │   Extraction    │     │  Calculator  │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────────────┘
                          │                      │                      │
                          ▼                      ▼                      ▼
                      .md file              .json file            Final Score
```

## ✨ Features

- 🔍 **高精度 OCR**：集成专用 VLM-OCR 模型（FireRed），精准处理复杂排版的证书/论文扫描件
- 🤖 **智能信息提取**：基于 Qwen3-VL-4B-Instruct 的语言理解能力，自动修复 OCR 乱序、断行等问题
- 📋 **多类型支持**：覆盖五大类证书场景
  - 🏅 竞赛获奖证书（Competition）
  - 💻 软件著作权（Software Copyright）
  - 📜 专利证书（Patent）
  - 📄 学术论文（Paper）
  - 🎖️ 荣誉称号（Honorary Title）
- ⚡ **一键 Pipeline**：单条命令完成全流程处理
- 📊 **结构化输出**：规范化 JSON 格式，便于后续对接加分系统

## 🛠️ Installation

### Prerequisites

- Python >= 3.8
- CUDA >= 11.8 (GPU 推理)
- vLLM 推理框架

### Clone Repository

```bash
git clone https://github.com/your-username/AutoCert.git
cd AutoCert
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download Models

| Model | Description | Download |
|-------|-------------|----------|
| FireRed-OCR | 专用证书 OCR 模型 | [🤗 HuggingFace](https://huggingface.co/FireRedTeam/FireRed-OCR) |
| Qwen3-VL-4B-Instruct | 信息提取 LLM | [🤗 HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |

## 🚀 Quick Start

### Option 1: Full Pipeline (Recommended)

一键运行完整处理流程：

```bash
cd /path/to/AutoCert

bash pipeline.sh \
    -v <version_tag> \
    -i /path/to/your/certificate_images \
    > logs/pipeline_$(date +%Y%m%d_%H%M%S).log 2>&1
```

**参数说明：**
| Parameter | Description |
|-----------|-------------|
| `-v` | 版本标签，用于区分不同批次的处理结果 |
| `-i` | 输入图片目录路径 |

**示例：**
```bash
bash pipeline.sh \
    -v batch_2024_spring \
    -i ./test_images \
    > logs/pipeline_$(date +%Y%m%d_%H%M%S).log 2>&1
```

---

### Option 2: Step-by-Step Execution

#### Step 1: OCR Recognition

使用 FireRed-OCR 将证书图片转换为 Markdown 文本：

```bash
cd /path/to/AutoCert

bash run_sequential.sh \
    -c run_seq_infer.txt \
    -o logs/firered-ocr_infer.log
```

#### Step 2: Structured Information Extraction

使用 Qwen3-VL-4B 从 Markdown 提取结构化信息：

```bash
cd /path/to/AutoCert

python qwen3_multi_vllm_infer-extract.py \
    --model_dir     /path/to/Qwen3-VL-4B-Instruct \
    --processor_dir /path/to/Qwen3-VL-4B-Instruct \
    --input_dir     ./results/firered_ocr/<ocr_output_folder> \
    --output_dir    ./results/qwen_extract/<extract_output_folder> \
    > logs/qwen_extract/run.log 2>&1
```

**参数说明：**
| Parameter | Description |
|-----------|-------------|
| `--model_dir` | Qwen3-VL-4B-Instruct 模型路径 |
| `--processor_dir` | 模型 Processor 路径（通常与 model_dir 相同） |
| `--input_dir` | OCR 阶段输出的 .md 文件目录 |
| `--output_dir` | 结构化 JSON 输出目录 |

## 📁 Pipeline

### Directory Structure

```
AutoCert/
├── 📂 test_images/              # 输入：证书图片
│   ├── cert_001.jpg
│   ├── cert_002.png
│   └── ...
├── 📂 results/
│   ├── 📂 firered_ocr/          # OCR 输出 (.md)
│   │   └── <version>/
│   │       ├── cert_001.md
│   │       └── cert_002.md
│   └── 📂 qwen_extract/         # 提取输出 (.json)
│       └── <version>/
│           ├── cert_001.json
│           └── cert_002.json
├── 📂 logs/                     # 运行日志
├── 📄 pipeline.sh               # 一键运行脚本
├── 📄 run_sequential.sh         # OCR 运行脚本
├── 📄 qwen3_multi_vllm_infer-extract.py  # 信息提取脚本
└── 📄 README.md
```

### Processing Flow

```mermaid
graph LR
    A[📷 Certificate Image] --> B[FireRed-OCR]
    B --> C[📝 Markdown Text]
    C --> D[Qwen3-VL-4B]
    D --> E[📋 Structured JSON]
    E --> F[🧮 Score Calculator]
    F --> G[🎯 Final Score]
```

## 📊 Output Format

### JSON Schema

提取的结构化信息输出为标准 JSON 格式：

```json
{
  "certificate_type": "Competition | Software Copyright | Patent | Paper | Honorary Title",
  "name": "证书/论文/专利的完整名称",
  "students": ["学生1", "学生2", "..."],
  "advisors": ["指导老师1", "指导老师2"],
  "team_name": "团队名称 | null",
  "issue_date": "YYYY-MM-DD | YYYY-MM | YYYY",
  "issuing_authority": ["颁发机构1", "颁发机构2"],
  "award_level": "一等奖 | 二等奖 | ... | null"
}
```

### Field Description

| Field | Type | Description |
|-------|------|-------------|
| `certificate_type` | `string` | 证书类型（五选一） |
| `name` | `string` | 完整名称，含赛事阶段、赛项等 |
| `students` | `array` | 学生/作者列表，保持原始顺序 |
| `advisors` | `array` | 指导老师列表，无则为空数组 |
| `team_name` | `string \| null` | 参赛团队名称 |
| `issue_date` | `string \| null` | 颁发/发表日期 |
| `issuing_authority` | `array` | 颁发机构列表 |
| `award_level` | `string \| null` | 获奖等级 |

### Examples

<details>
<summary>🏅 竞赛证书示例</summary>

```json
{
  "certificate_type": "Competition",
  "name": "2024年第十八届CIMC"西门子杯"中国智能制造挑战赛全国总决赛（智能制造工程设计与应用类赛项：信息化网络化方向（本科组））",
  "students": ["陈亚龙"],
  "advisors": [],
  "team_name": null,
  "issue_date": "2024",
  "issuing_authority": ["中国智能制造挑战全国总决赛组委会"],
  "award_level": "二等奖"
}
```
</details>

<details>
<summary>📄 学术论文示例</summary>

```json
{
  "certificate_type": "Paper",
  "name": "Knowledge Embedding With Graph Convolutional Network and Bidirectional Gated Recurrent Unit for Fault Diagnosis",
  "students": ["Jie Dong", "Cuiping Chen", "Chi Zhang", "Jinyao Ma", "Kaixiang Peng"],
  "advisors": [],
  "team_name": null,
  "issue_date": "2025-01-17",
  "issuing_authority": [],
  "award_level": null
}
```
</details>

## 🧠 Model Details

### Information Extraction Capabilities

系统使用 Qwen3-VL-4B-Instruct 的纯语言理解能力，具备以下智能预处理特性：

| Capability | Description |
|------------|-------------|
| **逆序修复** | 自动识别并重组 OCR 乱序文本块 |
| **断行合并** | 智能拼接被错误换行分割的词汇 |
| **噪声清洗** | 去除论文作者上标、机构标注等干扰 |
| **语义重建** | 基于上下文推断完整竞赛名称及赛项信息 |

### Supported Certificate Types

| Type | Chinese | Examples |
|------|---------|----------|
| Competition | 竞赛 | 数学建模、ACM、机器人大赛等 |
| Software Copyright | 软件著作权 | 计算机软件著作权登记证书 |
| Patent | 专利 | 发明专利、实用新型、外观设计 |
| Paper | 论文 | SCI/EI 论文、会议论文 |
| Honorary Title | 荣誉称号 | 三好学生、优秀学生干部等 |

## 🗺️ Roadmap

- [x] FireRed-OCR 证书识别模块
- [x] Qwen3-VL-4B 信息提取模块
- [x] 一键 Pipeline 脚本
- [ ] 📊 证书-加分对照表配置模块
- [ ] 🧮 自动加分计算引擎
- [ ] 🌐 Web UI 界面
- [ ] 📈 批量统计报表导出
- [ ] 🔗 教务系统 API 对接

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Qwen3-VL](https://github.com/QwenLM/Qwen3) - 强大的多模态大语言模型
- [vLLM](https://github.com/vllm-project/vllm) - 高性能 LLM 推理引擎
- FireRed-OCR - 专用文档 OCR 模型

## 📮 Contact

如有问题或建议，欢迎提交 [Issue](https://github.com/your-username/AutoCert/issues) 或 [Pull Request](https://github.com/your-username/AutoCert/pulls)。

---

<div align="center">

**如果这个项目对你有帮助，请给一个 ⭐ Star！**

Made with ❤️ for Graduate Recommendation

</div>
