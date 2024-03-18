本项目使用 [PyGCL](https://github.com/PyGCL/PyGCL)，论文链接[our paper](https://arxiv.org/abs/2109.01116)，
对[ImGCL](https://arxiv.org/pdf/2205.11332.pdf)进行了复现，后续将加入其他的小改进。

项目使用的库版本信息如下：

##Version Infomation

- python==3.9
- torch==2.2.0
- torch-cluster==1.6.3+pt22cu121
- torch-geometric==2.5.0
- torch-scatter==2.1.2+pt22cu121
- torch-sparse==0.6.18+pt22cu121
- torch-spline-conv==1.2.2+pt22cu121


项目结果如下。

\renewcommand{\arraystretch}{1} %控制行高
| Method           | WikiCS ImGCL | WikiCS ImGCL+TL | Amazon-Computers ImGCL | Amazon-Computers ImGCL+TL | Amazon-Photo ImGCL | Amazon-Photo ImGCL+TL |
|------------------|--------------|-----------------|------------------------|---------------------------|--------------------|-----------------------|
| 无权重           | 0.7876       | 0.7864          | 0.8394                 | 0.8752                    | 0.9049             | 0.9213                |
| 无权重+加噪      | 0.7955       | **0.8065**      | 0.8425                 | 0.8685                    | 0.9026             | **0.9304**            |
| 权重             | 0.7851       | 0.7803          | 0.8383                 | **0.8774**                | 0.9092             | 0.9150                |
| 权重+加噪        | 0.7889       | 0.7862          | 0.8308                 | 0.8755                    | 0.9082             | 0.9223                |


