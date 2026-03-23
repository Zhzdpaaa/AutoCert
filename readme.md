# 启动 FireRed_OCR 识别
cd /mnt/tidal-alsh01/dataset/OCRData/page_label_project/AutoCert
bash run_sequential.sh -c run_seq_infer.txt -o /mnt/tidal-alsh01/dataset/OCRData/page_label_project/AutoCert/logs/firered-ocr_test_infer.log
# 启动 Qwen3_4B 结构化信息
cd /mnt/tidal-alsh01/dataset/OCRData/page_label_project/AutoCert
python qwen3_multi_vllm_infer-extract.py \
    --model_dir     /mnt/tidal-alsh01/dataset/OCRData/public_models/Qwen3-VL-4B-Instruct \
    --processor_dir /mnt/tidal-alsh01/dataset/OCRData/public_models/Qwen3-VL-4B-Instruct \
    --input_dir     /mnt/tidal-alsh01/dataset/OCRData/page_label_project/AutoCert/results/firered_ocr/red_2b_test \
    --output_dir    /mnt/tidal-alsh01/dataset/OCRData/page_label_project/AutoCert/results/qwen_extract/test_1 \
    > /mnt/tidal-alsh01/dataset/OCRData/page_label_project/AutoCert/logs/qwen_extract/run.log 2>&1

# pipeline 
cd /mnt/tidal-alsh01/dataset/OCRData/page_label_project/AutoCert
bash pipeline.sh \
    -v test_1111111 \
    -i /mnt/tidal-alsh01/dataset/OCRData/page_label_project/AutoCert/test_images \
    > /mnt/tidal-alsh01/dataset/OCRData/page_label_project/AutoCert/logs/pipeline_$(date +%Y%m%d_%H%M%S).log 2>&1
