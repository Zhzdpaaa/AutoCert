import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import os
from tqdm import tqdm
from dataclasses import asdict
from PIL import Image
from transformers import AutoProcessor
import argparse
import torch

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

PROMPT = r'''You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

            1. Text Processing:
            - Accurately recognize all text content in the PDF image without guessing or inferring.
            - Convert the recognized text into Markdown format.
            - Maintain the original document structure, including headings, paragraphs, lists, etc.

            2. Mathematical Formula Processing:
            - Convert all mathematical formulas to LaTeX format.
            - Enclose inline formulas with,(,). For example: This is an inline formula,( E = mc^2,)
            - Enclose block formulas with,\[,\]. For example:,[,frac{-b,pm,sqrt{b^2 - 4ac}}{2a},]

            3. Table Processing:
            - Convert tables to HTML format.
            - Wrap the entire table with <table> and </table>.

            4. Figure Handling:
            - Ignore figures content in the PDF image. Do not attempt to describe or convert images.

            5. Output Format:
            - Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
            - For complex layouts, try to maintain the original document's structure and format as closely as possible.

            Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.
            '''

USER_PROMPT = r"Please convert this image to Markdown format following the instructions."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--processor_dir", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def collect_images(input_dir):
    images = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                images.append(os.path.join(root, name))
    return images


def split_list(data, n):
    """均匀切分 list 到 n 份"""
    return [data[i::n] for i in range(n)]


def worker(
    rank,
    gpu_id,
    image_paths,
    model_dir,
    processor_dir,
    output_dir,
    input_dir,   # 👈 新增
):
    # ⚠️ 必须在 import vllm 之前设置
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from vllm import LLM, EngineArgs, SamplingParams

    print(f"[Worker {rank}] Using GPU {gpu_id}, images: {len(image_paths)}")

    processor = AutoProcessor.from_pretrained(processor_dir)

    engine_args = EngineArgs(
        model=model_dir,
        max_model_len=32768,
        max_num_seqs=5,
        limit_mm_per_prompt={"image": 50},
        
    )
    engine_args = asdict(engine_args) | {"seed": 0}

    llm = LLM(**engine_args)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
    )

    for image_path in tqdm(image_paths, desc=f"GPU {gpu_id}"):
        # basename = os.path.splitext(os.path.basename(image_path))[0]
        # markdown_file = os.path.join(output_dir, f"{basename}.md")

        # 相对于 input_dir 的路径
        rel_path = os.path.relpath(image_path, input_dir)
        rel_path_no_ext = os.path.splitext(rel_path)[0]

        # 输出 markdown 路径（保持目录结构）
        markdown_file = os.path.join(output_dir, rel_path_no_ext + ".md")

        # 确保目录存在
        os.makedirs(os.path.dirname(markdown_file), exist_ok=True)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": PROMPT},
                ],
            },
        ]

        # messages = [
        #     {
        #         "role": "system",
        #         "content": PROMPT,
        #     },
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image", "image": image_path},
        #             {"type": "text", "text": USER_PROMPT},
        #         ],
        #     },
        # ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        outputs = llm.generate(
            {
                "prompt": inputs,
                "multi_modal_data": {"image": [Image.open(image_path)]}
            },
            sampling_params=sampling_params
        )

        text = outputs[0].outputs[0].text

        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(text)


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = collect_images(args.input_dir)
    assert len(image_paths) > 0, "No images found"

    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No CUDA devices found"

    print(f"Detected {num_gpus} GPUs, total images: {len(image_paths)}")

    chunks = split_list(image_paths, num_gpus)

    processes = []
    for rank, (gpu_id, chunk) in enumerate(zip(range(num_gpus), chunks)):
        p = mp.Process(
            target=worker,
            args=(
                rank,
                gpu_id,
                chunk,
                args.model_dir,
                args.processor_dir,
                args.output_dir,
                args.input_dir,   # 👈 新增
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
