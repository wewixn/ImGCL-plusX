本项目使用 [PyGCL](https://github.com/PyGCL/PyGCL)，论文链接[our paper](https://arxiv.org/abs/2109.01116)，
对[ImGCL](https://arxiv.org/pdf/2205.11332.pdf)进行了复现，后续将加入其他的小改进。


## Version Infomation
可以使用以下命令安装大多数需要的包。其中torch、torchvision、dgl建议手动安装。torch-cluster、torch-scatter、torch-sparse、torch-spline-conv需要手动安装，建议在[torch_geometric-whl](https://pytorch-geometric.com/whl/)网站获取whl文件，再使用pip安装，以匹配各个包的版本。torch与torchvision也可以在[torch-whl](https://download.pytorch.org/whl/)网站查找符合的版本。

```
pip install -r requirements.txt
```
dgl 2.0.0版本、torch与torchvision可以使用以下命令安装：
```
pip install dgl==2.0.0 -f https://data.dgl.ai/wheels/repo.html
pip install torch==2.2.0+cu121 -f https://download.pytorch.org/whl/torch/
pip install torchvision==0.17.0+cu121 -f https://download.pytorch.org/whl/torchvision/
```
torch-cluster、torch-scatter、torch-sparse、torch-spline-conv手动安装：
```
pip install torch-cluster torch-scatter torch-sparse torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.2.0%2Bcu121.html
```



项目使用的库版本信息如下：
- python==3.9
- torch==2.2.0+cu121
- torch-geometric==2.5.0
- torch-cluster==1.6.3+pt22cu121
- torch-scatter==2.1.2+pt22cu121
- torch-sparse==0.6.18+pt22cu121
- torch-spline-conv==1.2.2+pt22cu121

## Result(Accuracy)

项目结果如下。

| 数据集/方法       | 无权重 | 无权重+加噪 | 权重 | 权重+加噪 |
|------------------|--------|-------------|------|-----------|
|                  |        |             |      |           |
| ---------------- | ------ | ----------- | ---- | --------- |
| **WikiCS**       |        |             |      |           |
| ImGCL            | 0.7876 | **0.8065**  | 0.7851 | 0.7889  |
| ImGCL+TL         | 0.7864 | 0.8685      | 0.7803 | 0.7862  |
|                  |        |             |      |           |
| **Amazon-Computers** |        |         |      |           |
| ImGCL            | 0.8394 | 0.8425      | 0.8383 | 0.8308  |
| ImGCL+TL         | 0.8752 | 0.8685      | **0.8774** | 0.8755 |
|                  |        |             |      |           |
| **Amazon-Photo** |        |             |      |           |
| ImGCL            | 0.9049 | 0.9026      | 0.9092 | 0.9082  |
| ImGCL+TL         | 0.9213 | **0.9304**  | 0.9150 | 0.9223  |
