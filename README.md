# Jittor 手写数字生成


![主要结果](https://github.com/thenot123/jittor/tree/main/result/result.png)



## 简介

本项目包含了第二届计图挑战赛计图 - 手写数字生成的代码实现。本项目的特点是：采用了生成对抗网络方法对MNIST数据集进行训练，训练好的模型可以较好的生成想要的手写数字编号。

## 安装

本项目可在 1 张 3090 上运行，训练时间约为10分钟。

#### 运行环境
- Centos7
- python >= 3.9
- jittor >= 1.3.0

#### 安装依赖
执行以下命令安装 python 依赖。
```
pip install -r requirements.txt
```

#### 预训练模型
预训练模型在model文件夹内。


## 训练
训练可运行以下命令：
```
python CGAN.py
```


## 推理


生成测试集上的结果可以运行以下命令：

```
python CGAN_test.py
```

