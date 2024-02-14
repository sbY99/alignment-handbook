CUDA_VISIBLE_DEVICES=0,1,2 python scripts/inference.py \
--model_name=GAI-LLM/Yi-Ko-6B-mixed-v15 \
--adapter_path=model/GAI-LLM-Yi-Ko-6B-mixed-v15-sft-qlora-dpo-v4 \
--max_length=512 \
--output_path=result/GAI-LLM-Yi-Ko-6B-mixed-v15-qlora-dpo-v4.csv \
--response_path=result/GAI-LLM-Yi-Ko-6B-mixed-v15-qlora-dpo-v4.txt 