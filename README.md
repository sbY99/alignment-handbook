## Install
```
conda create -n llm_train python=3.10
```
```
conda activate llm_train
```
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
```
python -m pip install .
```

## Train
```
sh run_sft_lora.sh
```

## Inference
```
sh inference_lora.sh
```

## Citation

If you find the content of this repo useful in your work, please cite it as follows:

```bibtex
@misc{alignment_handbook2023,
  author = {Lewis Tunstall and Edward Beeching and Nathan Lambert and Nazneen Rajani and Shengyi Huang and Kashif Rasul and Alexander M. Rush and Thomas Wolf},
  title = {The Alignment Handbook},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/alignment-handbook}}
}
```
