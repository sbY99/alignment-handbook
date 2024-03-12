CUDA_VISIBLE_DEVICES=1 nohup python scripts/inference.py \
--model_name=yanolja/EEVE-Korean-10.8B-v1.0 \
--adapter_path=model/yanolja-EEVE-Korean-10.8B-v1.0-v5-default-chat-template \
--is_adapter_model=True \
--max_new_tokens=512 \
--output_path=result/yanolja-EEVE-Korean-10.8B-v1.0-v5-default-chat-template.csv \
--response_path=result/yanolja-EEVE-Korean-10.8B-v1.0-v5-default-chat-template.json > inference_v2.out &