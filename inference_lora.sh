CUDA_VISIBLE_DEVICES=0,1,2 python scripts/inference.py \
--model_name=model/GAI-LLM-Yi-Ko-6B-mixed-v15-sft-qlora-v4 \
--is_adapter_model=True \
--max_length=512 \
--output_path=result/GAI-LLM-Yi-Ko-6B-mixed-v15-qlora-v4.csv \
--response_path=result/GAI-LLM-Yi-Ko-6B-mixed-v15-qlora-v4.txt 