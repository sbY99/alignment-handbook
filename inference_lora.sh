CUDA_VISIBLE_DEVICES=1 nohup python scripts/inference.py \
--model_name=yanolja/EEVE-Korean-10.8B-v1.0 \
--adapter_path=model/final-model \
--test_data_path=raw-data/test.csv \
--submission_data_path=raw-data/sample_submission.csv \
--max_new_tokens=512 \
--output_path=result/output.csv \
--response_path=result/output-text.json > inference.out &