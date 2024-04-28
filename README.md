本项目使用 [PyGCL](https://github.com/PyGCL/PyGCL)，论文链接[our paper](https://arxiv.org/abs/2109.01116)，
对[ImGCL](https://arxiv.org/pdf/2205.11332.pdf)进行了复现，后续将加入其他的小改进。


## Version Infomation
可以使用以下命令安装需要的包。其中torch、torchvision、dgl建议手动安装。torch-cluster、torch-scatter、torch-sparse、torch-spline-conv需要手动安装，建议在[torch_geometric-whl](https://pytorch-geometric.com/whl/)网站获取whl文件，再使用pip安装，以匹配各个包的版本。torch与torchvision也可以在[torch-whl](https://download.pytorch.org/whl/)网站查找符合的版本。

```
pip install torch==2.2.0+cu121 -f https://download.pytorch.org/whl/torch/
pip install torchvision==0.17.0+cu121 -f https://download.pytorch.org/whl/torchvision/
pip install dgl==2.0.0 -f https://data.dgl.ai/wheels/repo.html
pip install torch-cluster torch-scatter torch-sparse torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.2.0%2Bcu121.html
pip install -r requirements.txt
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

<table>
  <tr>
    <th rowspan="2">数据集/方法</th>
    <th colspan="2">base(GBT)</th>
    <th colspan="2">PSGCL(ours)</th>
  </tr>
  <tr>
    <td>micro-F1</td>
    <td>macro-F1</td>
    <td>micro-F1</td>
    <td>macro-F1</td>
  </tr>
  <tr>
    <td><strong>WikiCS</strong></td>
    <td>0.7884</td>
    <td>0.7612</td>
    <td>0.8010</td>
    <td>0.7700</td>
  </tr>
  <tr>
    <td><strong>Amazon-Photo</strong></td>
    <td>0.9163</td>
    <td>0.8953</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td><strong>cora</strong></td>
    <td>0.7787</td>
    <td>0.7697</td>
    <td>0.8219</td>
    <td>0.8131</td>
  </tr>
</table>