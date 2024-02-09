CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/inference.py \
--model_name=GAI-LLM/Yi-Ko-6B-mixed-v15 \
--adapter_path=model/GAI-LLM-Yi-Ko-6B-mixed-v15-sft-qlora-v1 \
--max_length=512 \
--output_path=result/GAI-LLM-Yi-Ko-6B-mixed-v15-qlora-v1-2.csv \
--response_path=result/GAI-LLM-Yi-Ko-6B-mixed-v15-qlora-v1-2.txt 