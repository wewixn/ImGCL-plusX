
复现相关问题：项目使用python3.7.4。
原作者环境配置不太准确，
推荐使用以下命令配置环境。
```
pip install torch==1.7.0+cu110 -f https://download.pytorch.org/whl/torch/
pip install torch_geometric==1.7.0
pip install torch-cluster==1.5.9 torch-scatter==2.0.7 torch-sparse==0.6.9 torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.7.0%2Bcu110.html
pip install pyyaml ogb
```
同时使用conda或者apt-get获取cuda toolkit 11.0
```
apt-get install -y cuda-toolkit-11-0
```
对本文加入的PSGCL方法，还需要安装imblearn库。
同时imblearn对python3.7的支持不是很好，安装后报错部分把imblearn对应的 '/' 删除即可运行。
```
pip install imblearn
```

以下是原作者的README.md

---

# ProGCL: Rethinking Hard Negative Mining in Graph Contrastive Learning (ICML 2022)
PyTorch implementation for [ProGCL: Rethinking Hard Negative Mining in Graph Contrastive Learning](https://arxiv.org/abs/2110.02027) accepted by ICML 2022.
## Requirements
* Python 3.7.4
* PyTorch 1.7.0
* torch_geometric 1.5.0
* tqdm
## Training & Evaluation
ProGCL-weight:
```
python train.py --device cuda:0 --dataset Amazon-Computers --param local:amazon-computers.json --mode weight
```
ProGCL-mix:
```
python train.py --device cuda:0 --dataset Amazon-Computers --param local:amazon-computers.json --mode mix
```
## Useful resources for Pretrained Graphs Neural Networks
* The first comprehensive survey on this topic: [A Survey of Pretraining on Graphs: Taxonomy, Methods, and Applications](https://arxiv.org/abs/2202.07893v1)
* [A curated list of must-read papers, open-source pretrained models and pretraining datasets.](https://github.com/junxia97/awesome-pretrain-on-graphs)

## Citation
```
@inproceedings{xia2022progcl,
  title={ProGCL: Rethinking Hard Negative Mining in Graph Contrastive Learning},
  author={Xia, Jun and Wu, Lirong and Wang, Ge and Li, Stan Z.},
  booktitle={International conference on machine learning},
  year={2022},
  organization={PMLR}
}
```
