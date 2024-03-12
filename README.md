## Init
### If you downloaded all the code locally, skip it.
```
git clone https://github.com/sbY99/alignment-handbook.git
cd alignment-handbook/
```

## Install
```
conda create -n llm_train python=3.10
conda activate llm_train
```

### Install torch
- Please refer: (https://pytorch.org/get-started/previous-versions/)
```
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Install packages for training
```
python -m pip install .
pip install sentence_transformers
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
### You have to specify your GPU number.
### Required Files: 
- data/eval.csv (download using the 'Download Dataset' command)
- data/sample_submission.csv (download using the 'Download Dataset' command)
### Save the model weight in model/final-model
```
sh run_sft_lora.sh
```

## Inference

### You have to specify your GPU number.
### Required Files: 
- model/final-model (train the model using above train command or download the model weights)
- data/eval.csv (download using the 'Download Dataset' command)
- data/sample_submission.csv (download using the 'Download Dataset' command)
  
### There are two output files. 
- result/output.csv: file used for dacon submission that the model answer is embedded.
- result/output-text.json: file that allows you to check answers in text form, consisting of questions and answers.
```
sh inference_lora.sh
```

## Citation

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
