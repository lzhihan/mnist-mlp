# MNIST 手写数字识别 - 多层感知机 (MLP)

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 目录

- [项目概述](#项目概述)
- [主要特点](#主要特点)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [模型架构](#模型架构)
- [训练配置](#训练配置)
- [使用指南](#使用指南)
- [结果展示](#结果展示)
- [常见问题](#常见问题)
- [扩展建议](#扩展建议)
- [学习资源](#学习资源)
- [贡献](#贡献)
- [许可证](#许可证)

## 项目概述

本项目实现了一个基于**多层感知机 (MLP)** 的 MNIST 手写数字识别系统，采用 PyTorch 框架。MNIST 数据集包含 70,000 张 28x28 像素的手写数字图像（0-9），本项目通过构建全连接神经网络进行高效训练和预测。

### 核心功能

✅ **完整训练流程**：包含数据加载、模型训练、验证评估、早停机制、学习率调度  
✅ **图形化预测界面**：提供直观的绘制板，支持实时手绘数字并查看预测结果  
✅ **命令行预测**：支持从图片文件批量预测  
✅ **可视化分析**：自动生成训练曲线图，直观展示损失和准确率变化  
✅ **最佳模型保存**：自动保存测试准确率最高的模型权重

---

## 🚀 主要特点

- **高效训练**：使用 Adam 优化器和自适应学习率调整
- **GPU 加速**：自动检测并使用 CUDA 设备进行训练和推理
- **早停机制**：防止过拟合，自动在性能不再提升时停止训练
- **用户友好 GUI**：简洁直观的界面，支持鼠标绘制数字
- **智能预处理**：自动裁剪、缩放、居中，模拟 MNIST 数据分布
- **实时反馈**：GUI 中显示概率分布条，直观了解模型置信度

---

## 📦 项目结构

```
mnist_mlp/
├── best_mlp_mnist.pth          # 训练好的模型权重文件（约 0.8MB）
├── requirements.txt            # Python 依赖包列表
├── train_mlp.py                # 训练脚本（含模型定义）
├── predict_gui.py              # GUI 预测工具（版本 1：基础版）
├── predict_gui_2.py            # GUI 预测工具（版本 2：增强预览版）
├── .gitignore                  # Git 忽略文件配置
├── README.md                   # 项目说明文档
├── AGENTS.md                   # AI 开发指南
├── data/                       # MNIST 数据集目录（自动下载）
│   └── MNIST/
│       ├── raw/                # 原始数据
│       └── processed/          # 处理后数据
├── results/                    # 训练结果输出目录
│   └── training_curves.png     # 训练曲线可视化图
└── __pycache__/                # Python 字节码缓存
```

### 文件说明

| 文件 | 大小 | 描述 |
|------|------|------|
| `train_mlp.py` | ~297 行 | 完整的训练流程，包括数据加载、模型定义、训练循环、可视化 |
| `predict_gui.py` | ~200 行 | Tkinter 图形界面，支持手绘预测 |
| `best_mlp_mnist.pth` | ~0.8MB | 预训练模型权重，可直接用于预测 |

---

## 🏗️ 模型架构

### MLP 网络结构

```
输入层 (784)  →  隐藏层 (256)  →  输出层 (10)
   ↓               ↓              ↓
展平图像        ReLU 激活      Softmax
```

#### 详细参数

| 层级 | 类型 | 输入维度 | 输出维度 | 激活函数 |
|------|------|----------|----------|----------|
| 输入层 | Flatten | (1, 28, 28) | 784 | - |
| 隐藏层 1 | Linear | 784 | 256 | ReLU |
| 输出层 | Linear | 256 | 10 | Softmax |

**参数量计算**：
- FC1: 784 × 256 + 256 = 200,960
- FC2: 256 × 10 + 10 = 2,570
- **总计**: ~203,530 个可训练参数

### 前向传播流程

1. **输入展平**：将 28×28 图像重塑为 784 维向量
2. **第一层变换**：`z₁ = W₁·x + b₁`
3. **ReLU 激活**：`a₁ = max(0, z₁)`
4. **第二层变换**：`z₂ = W₂·a₁ + b₂`
5. **Softmax 输出**：`pᵢ = exp(z₂ᵢ) / Σⱼexp(z₂ⱼ)`

### 为什么选择 MLP？

✅ **简单高效**：适合 MNIST 这类相对简单的任务  
✅ **易于理解**：代码清晰，便于学习神经网络基础  
✅ **训练快速**：相比 CNN，参数量少，训练速度快  
⚠️ **局限性**：忽略空间结构信息，不适合复杂图像任务

---

### 什么是 MLP？

多层感知机 (Multilayer Perceptron, MLP) 是一种前馈人工神经网络，由多个层组成：输入层、一个或多个隐藏层、输出层。每层由多个神经元（节点）组成，相邻层之间通过权重连接。

MLP 是深度学习的基础模型之一，用于解决分类和回归问题。在本项目中，MLP 用于将 28x28 的图像（784 个像素）分类为 10 个数字类别（0-9）。

### MLP 的基本结构

1. **输入层**：接收输入数据。在 MNIST 中，输入是 784 个像素值（展平后的 28x28 图像）。
2. **隐藏层**：执行计算，提取特征。本项目使用一个隐藏层，包含 128 个神经元。
3. **输出层**：产生最终输出。对于分类，输出层有 10 个神经元，对应 10 个类别。

### 前向传播 (Forward Propagation)

前向传播是从输入到输出的计算过程：

1. **线性变换**：每个神经元计算输入的加权和，加上偏置。
   ```
   z = W * x + b
   ```
   - `W` 是权重矩阵
   - `x` 是输入向量
   - `b` 是偏置向量
   - `z` 是线性输出

2. **激活函数**：引入非线性，使网络能学习复杂模式。
   - 常用激活函数：ReLU (Rectified Linear Unit)
     ```
     a = max(0, z)
     ```
   - ReLU 简单高效，避免梯度消失问题。

3. **输出层**：使用 Softmax 激活函数，将输出转换为概率分布。
   ```
   softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)
   ```

在本项目中：
- 输入：784 维向量
- 隐藏层：128 个神经元，使用 ReLU 激活
- 输出：10 维向量，使用 Softmax

### 反向传播 (Backpropagation)

反向传播是训练 MLP 的关键，用于更新权重和偏置，减少预测误差。

1. **损失函数**：衡量预测与真实标签的差距。
   - 分类问题使用交叉熵损失 (Cross-Entropy Loss)：
     ```
     L = -sum(y_true * log(y_pred))
     ```
     - `y_true` 是真实标签（one-hot 编码）
     - `y_pred` 是预测概率

2. **梯度计算**：使用链式法则计算损失对每个参数的梯度。
   - 从输出层向输入层反向传播误差。

3. **优化算法**：更新参数以最小化损失。
   - 本项目使用 Adam 优化器，结合动量和自适应学习率。
   - 更新公式：
     ```
     W = W - learning_rate * gradient
     ```

### 训练过程

1. **数据准备**：加载 MNIST 数据集，分为训练集和测试集。
2. **前向传播**：计算预测。
3. **计算损失**：比较预测与标签。
4. **反向传播**：计算梯度。
5. **更新参数**：使用优化器调整权重和偏置。
6. **重复**：多个 epoch，直到收敛。

### 为什么 MLP 适合 MNIST？

- **简单有效**：MNIST 图像相对简单，MLP 能很好地学习像素到类别的映射。
- **可扩展**：隐藏层可以增加深度和宽度，学习更复杂特征。
- **易理解**：MLP 的工作原理直观，便于学习神经网络基础。

### 局限性

- **不适合复杂图像**：对于更大、更高分辨率的图像，MLP 参数过多，容易过拟合。卷积神经网络 (CNN) 更适合图像任务。
- **无空间信息**：MLP 忽略像素的空间关系，而 CNN 能捕捉局部特征。

## 🚀 快速开始

### 1. 环境要求

- **Python**: 3.7 或更高版本
- **PyTorch**: 1.9.0+
- **其他依赖**: NumPy, Matplotlib, Pillow, SciPy

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

如果还没有 `requirements.txt`，可以手动安装：
```bash
pip install torch>=1.9.0 torchvision>=0.10.0 numpy>=1.19.0 matplotlib>=3.3.0 pillow>=8.0.0 scipy>=1.5.0
```

### 3. 训练模型

```bash
python train_mlp.py
```

训练完成后会生成：
- `best_mlp_mnist.pth` - 最佳模型权重
- `results/training_curves.png` - 训练曲线图

### 4. 启动预测 GUI

```bash
python predict_gui.py
```

或使用增强预览版本：
```bash
python predict_gui_2.py
```

---

### 环境要求

- Python 3.7+
- PyTorch
- NumPy, Matplotlib, PIL (Pillow), SciPy

安装依赖：
```bash
pip install -r requirements.txt
```

### 训练模型（详细）

训练过程会自动：
1. 下载 MNIST 数据集到 `data/` 目录
2. 创建数据加载器
3. 初始化模型和优化器
4. 执行训练循环（最多 10 个 epoch）
5. 保存最佳模型
6. 生成可视化图表

**预期输出示例**：
```
当前使用设备：cuda
Epoch 1 [0/60000 (0.00%)]    Loss: 0.3845    ETA: 12.3s
...
Epoch 1 训练完成！平均损失：0.2543, 训练准确率：92.45% (耗时：45.23s)
测试集结果：平均损失：0.1234, 测试准确率：96.78% (5807/6000)
✓ 保存最佳模型 (准确率：96.78%)
...
训练结束！最高测试准确率：97.85%
训练曲线已保存到：./results/training_curves.png
```

---

### 预测

### GUI 模式（手绘预测）

启动图形界面后：

1. **绘制数字**：在左侧黑色画布上用鼠标绘制白色数字
2. **查看预览**：右侧实时显示预处理后的 28x28 图像（放大 8 倍）
3. **点击预测**：模型会输出预测结果和每个类别的概率
4. **清空重绘**：点击"清空"按钮清除当前内容

**界面元素**：
- 🖌️ **画布区域**：黑色背景，白色笔迹，支持流畅绘制
- 👁️ **预览区域**：显示预处理后的标准输入
- 📊 **概率条**：绿色进度条展示 10 个数字的预测概率
- 🔘 **功能按钮**：预测、清空、加载模型、保存图像

---

### 命令行模式（图片预测）

从现有图片文件进行预测：

```bash
python predict_gui.py --image path/to/image.png --model best_mlp_mnist.pth
```

**参数说明**：
- `--image`: 输入图片路径（支持 PNG、JPG 等格式）
- `--model`: 模型权重文件路径（默认为 `best_mlp_mnist.pth`）

**输出示例**：
```
预测结果：5
概率分布：
  0: 0.01%
  1: 0.05%
  ...
  5: 98.72%  ← 最高
  ...
```

---

## 使用指南

### GUI 界面

- **画布**：黑色背景，绘制白色数字。笔刷粗细固定，支持平滑绘制。
- **预览**：右侧显示预处理后的 28x28 图像（放大 8 倍）。
- **概率条**：显示每个数字的预测概率（绿色进度条）。
- **按钮**：
  - 预测：运行模型预测。
  - 清空：重置画布和结果。
  - 加载模型：选择自定义模型文件。
  - 保存图像：保存预处理图像。

### 预处理说明

绘制时，代码模拟 MNIST 预处理：
1. 裁剪最小包围框。
2. 缩放到 ~20x20，保持宽高比。
3. 居中填充到 28x28，保留灰度（不二值化）。

这确保输入与训练数据一致。

## ⚙️ 训练配置

### 超参数设置

```python
BATCH_SIZE = 64        # 批次大小
LEARNING_RATE = 0.001  # 初始学习率
EPOCHS = 10            # 最大训练轮数
HIDDEN_SIZE = 256      # 隐藏层神经元数量
PATIENCE = 3           # 早停耐心值
```

### 优化策略

- **优化器**：Adam（自适应矩估计）
- **学习率调度**：ReduceLROnPlateau（当验证准确率停滞时减半学习率）
- **早停机制**：连续 3 个 epoch 未提升则停止训练
- **最佳模型保存**：自动保存测试准确率最高的权重

### 数据处理

```python
transforms.Compose([
    transforms.ToTensor(),  # PIL 图像转 Tensor，归一化到 [0, 1]
    # 可选：标准化（已注释）
    # transforms.Normalize((0.1307,), (0.3081,))
])
```

---

## 📊 结果展示

### 训练曲线

训练完成后，自动生成 `results/training_curves.png`，包含：
- **左图**：训练损失随 epoch 的变化
- **右图**：训练准确率和测试准确率对比

### 典型性能

在 MNIST 测试集上的表现：
- **训练准确率**: 98-99%
- **测试准确率**: 97-98%
- **训练时间**: 约 5-10 分钟（CPU）/ 1-2 分钟（GPU）

### 混淆矩阵（可选扩展）

可以添加混淆矩阵来可视化各类别的分类情况。

---

## ❓ 常见问题

### Q1: 为什么我的准确率比较低？

**可能原因**：
1. 训练轮数不足（尝试增加 EPOCHS）
2. 学习率不合适（调整 LEARNING_RATE）
3. 隐藏层太小（增加 HIDDEN_SIZE）
4. 没有正确预处理输入图像

### Q2: GPU 和 CPU 训练有什么区别？

- **速度**：GPU 快 5-10 倍
- **准确率**：无差异（相同的算法）
- **显存需求**：本模型仅需约 10MB 显存

### Q3: 如何自定义模型架构？

修改 `train_mlp.py` 中的 `MLP` 类：
```python
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)  # 添加更多层
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
```

### Q4: GUI 绘制时识别不准确怎么办？

确保：
1. 绘制清晰、居中的数字
2. 数字大小适中（不要太大或太小）
3. 使用白色笔迹在黑色背景上
4. 加载正确的模型文件

### Q5: 如何在自己的数据集上训练？

1. 准备数据格式与 MNIST 一致（28x28 灰度图）
2. 修改 `create_data_loaders()` 中的数据路径
3. 调整 `num_classes` 参数
4. 重新训练模型

---

## 💡 扩展建议

如果你想进一步提升这个项目：

### 1. 改进模型

- **添加 Dropout**：防止过拟合
  ```python
  self.dropout = nn.Dropout(p=0.2)
  ```
- **多层隐藏**：增加网络深度
  ```python
  self.fc2 = nn.Linear(256, 128)
  self.fc3 = nn.Linear(128, 10)
  ```
- **批量归一化**：加速收敛
  ```python
  self.bn = nn.BatchNorm1d(hidden_size)
  ```

### 2. 数据增强

- 随机旋转（±10°）
- 轻微平移（10%）
- 弹性形变

### 3. 升级为 CNN

卷积神经网络（CNN）在图像任务上表现更好：
```python
conv1 = Conv2d(1, 32, 3)  # 28x28 -> 26x26
conv2 = Conv2d(32, 64, 3) # 12x12 -> 10x10
fc = Linear(64*7*7, 10)
```

### 4. 添加功能

- **批量预测**：一次性处理多张图片
- **Web 界面**：使用 Gradio 或 Streamlit
- **导出 ONNX**：部署到其他平台
- **移动端部署**：转换为 TorchScript 或 CoreML

### 5. 性能监控

- 添加 TensorBoard 可视化
- 记录详细的训练日志
- 生成混淆矩阵和分类报告

---

## 📚 学习资源

### 官方文档

- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [PyTorch 文档](https://pytorch.org/docs/stable/index.html)
- [TorchVision transforms](https://pytorch.org/vision/stable/transforms.html)

### 数据集

- [MNIST 官方网站](http://yann.lecun.com/exdb/mnist/)
- [Kaggle MNIST 竞赛](https://www.kaggle.com/c/digit-recognizer)

### 课程推荐

- [吴恩达机器学习](https://www.coursera.org/learn/machine-learning) - 经典入门
- [李宏毅深度学习](https://speech.ee.ntu.edu.tw/~hylee/ml/) - 中文讲解
- [Fast.ai 实战课程](https://course.fast.ai/) - 自上而下学习法

### 书籍推荐

- 《深度学习》（花书）- Ian Goodfellow
- 《动手学深度学习》- 阿斯顿·张等
- 《PyTorch 深度学习实战》

### 进阶阅读

- [Understanding MLPs](https://towardsdatascience.com/understanding-feedforward-neural-networks-and-backpropagation-2a8d7d6b4f5b)
- [Why CNNs > MLPs for Images](https://cs231n.github.io/convolutional-networks/)

---

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

### 贡献方式

1. **提交 Issue**：报告 Bug 或请求新功能
2. **Pull Request**：修复问题或添加特性
3. **改进文档**：补充说明、示例或翻译
4. **分享经验**：你的训练结果和使用心得

### 开发环境设置

```bash
# 克隆项目
git clone https://github.com/yourusername/mnist-mlp.git
cd mnist-mlp

# 安装依赖
pip install -r requirements.txt

# 安装开发工具（可选）
pip install pytest ruff black mypy
```

### 代码规范

请参考 [AGENTS.md](AGENTS.md) 中的代码风格规范。

---

## 许可证

MIT License
