## 安装

项目主要基于Python和PyTorch深度学习框架完成，所有依赖包罗列在项目文件`requirements.txt`中，可以通过如下命令安装

```shell
pip install -r requirements.txt
```

## 数据准备

```shell
─data
│  ├─figure_ground
│  └─rgb
─datatest
│  ├─figure_ground
│  └─rgb
```

训练数据默认放在`data`文件夹下，其中`figure_ground`子文件夹为标签图像，`rgb`为用于输入的原始三通道图像；测试数据默认存放在`datatest`文件夹下，可以在训练脚本文件和测试脚本文件最开始的超参数部分修改数据集路径位置

## 文件说明

- `train.py`
  - 用于进行模型的训练，训练超参数在文件最开始部分修改
  - 会在`./ckpt`文件夹下输出训练模型
- `test.py`
  - 用于进行模型的测试，训练超参数在文件最开始部分修改
  - 指标会在运行终端中输出，结果图像会保存至`./result`文件夹下
- `analyze.py`
  - `get_miou`函数用于计算mIoU指标
  - `boundary_iou`函数用于计算Boundary IoU指标
- `data.py`：完成数据的读取、预处理操作
- `FPN.py`：定义FPN特征金字塔模型
- `loss.py`：按照实验报告3.2节的描述计算损失函数，具体实现函数为`seg_loss`
- `unet.py`：定义U-net模型

## 训练与测试

- 训练

```shell
python train.py
```

- 测试

```shell
python test.py
```

## 结果总览

详细的结果指标以及分析可见实验报告，这里列出所采用模型的mIoU和Boundary IoU

| mIoU | Boundary IoU |
| ---- | ------------ |
|      |              |

