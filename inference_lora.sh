CUDA_VISIBLE_DEVICES=1,2 nohup python scripts/inference.py \
--model_name=yanolja/KoSOLAR-10.7B-v0.2 \
--adapter_path=model/yanolja-KoSOLAR-10.7B-v0.2-sft-qlora-v5 \
--is_adapter_model=True \
--max_new_tokens=512 \
--output_path=result/yanolja-KoSOLAR-10.7B-v0.2-sft-qlora-v5.csv \
--response_path=result/yanolja-KoSOLAR-10.7B-v0.2-sft-qlora-v5.json &