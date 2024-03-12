## Install
```
git clone https://github.com/sbY99/alignment-handbook.git
cd alignment-handbook/
```

```
conda create -n llm_train python=3.10
conda activate llm_train
```
```
# Install torch
# Please refer: https://pytorch.org/get-started/previous-versions/
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

```
# Packages for training
python -m pip install .
```

## Download Dataset
```
pip install gdown
mkdir data
```

```
export data_path=data
gdown https://drive.google.com/drive/folders/19_sIUa6wbpVpTRv232tygxU0-Au2z_GQ -O $data_path --folder
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
