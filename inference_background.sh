CUDA_VISIBLE_DEVICES=1,2,3 nohup python scripts/inference.py \
--model_name=LDCC/LDCC-SOLAR-10.7B \
--adapter_path=model/LDCC-SOLAR-10.7B-sft-qlora-v3 \
--max_length=512 \
--output_path=result/LDCC-SOLAR-10.7B-sft-qlora-v3.csv \
--response_path=result/LDCC-SOLAR-10.7B-sft-qlora-v3.txt &