复现相关问题：项目使用python3.9。原作者环境配置不太准确，
推荐使用以下命令配置环境。
```
pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch/
pip install dgl-cu113==0.8.2.post1 -f https://data.dgl.ai/wheels/repo.html
pip install torchvision==0.12.0+cu113  -f https://download.pytorch.org/whl/torchvision/
pip install torchaudio==0.11.0+cu113  -f https://download.pytorch.org/whl/torchaudio/
```

同时使用conda或者apt-get获取cuda toolkit 11.3，
并使用其安装torch-cluster、torch-scatter、torch-sparse、torch-spline-conv。
手动安装上述特定版本的库后，再安装其余版本宽松的库。
```
apt-get install -y cuda-toolkit-11-3
pip install torch-cluster==1.6.3 torch-scatter==2.1.2 torch-sparse==0.6.14 torch-spline-conv==1.2.2 -f https://pytorch-geometric.com/whl/torch-1.11.0%2Bcu113.html
pip install -r requirements.txt
```
对本文加入的PSGCL方法，还需要安装imblearn库。
```
pip install imblearn
```

以下是原作者的README.md

---

## Covariance-Preserving Feature Augmentation for Graph Contrastive Learning.

### Overview
This repo contains an example implementation for KDD'22 paper: **COSTA: Covariance-Preserving Feature Augmentation for Graph Contrastive Learning**. 
This code provides the multiview MV-COSTA. The SV-COSTA can be easily obtained by modifying the MV-COSTA.

### Overview

COSTA is a **feature augmentation** method that generates augmented samples in the feature space (latent space). It produced a bias-free and covariance-bounded augmentation to alleviate the bias problem in the typical graph augmentation (e.g., edge permutations). 

### Dependencies
Our implementation works with PyTorch>=1.0.0 Install other dependencies: `$ pip install -r requirement.txt`

### Reproduce our results
We provide several datasets to reproduce our results. We provide wandb logs to show the performance. See following to see the detail

https://wandb.ai/yifeiacc/COSTA_public?workspace=user-yifeiacc

The detailed settings (including hyper-parameters and GPUs) and the results can be found in these logs. You can directly checkout to the corresponding branch(commit).

### Usage
To run our code, just run the following
```
$ cd src 
$ python main.py --root path/to/COSTA/dir --dataset Cora --model COSTA --config COSTA_default.yaml
