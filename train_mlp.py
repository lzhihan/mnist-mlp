import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os

# --------------------------
# 1. 超参数设置
# --------------------------
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
HIDDEN_SIZE = 256  # 隐藏层神经元数量
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 3  # 早停耐心值

# 全局变量用于存储训练历史
train_losses = []
train_accuracies = []
test_accuracies = []


# --------------------------
# 2. 模型定义
# --------------------------
class MLP(nn.Module):
    """多层感知机模型"""

    def __init__(self, input_size=28 * 28, hidden_size=256, num_classes=10):
        super(MLP, self).__init__()
        # 第一层：输入 (784) -> 隐藏层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        # 第二层：隐藏层 -> 输出 (10)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        # 可选：添加 Dropout 防止过拟合 (这里暂时不加，因为 MLP 在 MNIST 上不容易过拟合)
        # self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # x 的形状初始为 [batch_size, 1, 28, 28]
        # 需要展平为 [batch_size, 784]
        x = x.view(-1, 28 * 28)

        out = self.fc1(x)
        out = self.relu(out)
        # out = self.dropout(out)

        out = self.fc2(out)
        # 注意：这里不直接加 Softmax，因为 nn.CrossEntropyLoss 内部包含了 Softmax
        return out


# --------------------------
# 3. 数据加载函数
# --------------------------
def create_data_loaders():
    """创建数据加载器"""
    # MNIST 图片是灰度图 (1 channel), 大小 28x28
    # transforms.ToTensor() 会将像素值从 [0, 255] 归一化到 [0.0, 1.0]
    transform = transforms.Compose([
        transforms.ToTensor(),
        # 可选：进一步标准化 (均值 0.1307, 标准差 0.3081 是 MNIST 的统计值)
        # transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 下载并加载训练集
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True  # 每个 epoch 打乱数据
    )

    # 下载并加载测试集
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1000,
        shuffle=False
    )

    return train_loader, test_loader


# --------------------------
# 4. 模型创建函数
# --------------------------
def create_model():
    """创建 MLP 模型、损失函数和优化器"""
    # 初始化模型
    input_size = 28 * 28
    num_classes = 10
    model = MLP(input_size, HIDDEN_SIZE, num_classes).to(DEVICE)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    return model, criterion, optimizer, scheduler


# --------------------------
# 5. 训练与测试函数
# --------------------------
def train(epoch, model, train_loader, criterion, optimizer):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        # 1. 前向传播
        optimizer.zero_grad()  # 清空梯度
        output = model(data)

        # 2. 计算损失
        loss = criterion(output, target)

        # 3. 反向传播
        loss.backward()
        optimizer.step()  # 更新权重

        total_loss += loss.item()

        # 计算当前 batch 的准确率
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        if batch_idx % 100 == 0:
            elapsed = time.time() - start_time
            batches_done = batch_idx + 1
            eta = elapsed / max(batches_done, 1) * (len(train_loader) - batches_done)
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.4f}\tETA: {eta:.1f}s')

    end_time = time.time()
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)

    print(
        f'\nEpoch {epoch} 训练完成！平均损失：{avg_loss:.4f}, 训练准确率：{accuracy:.2f}% (耗时：{end_time - start_time:.2f}s)')


def test(model, test_loader, criterion):
    """在测试集上评估模型"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # 测试时不需要计算梯度
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            test_loss += criterion(output, target).item()

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    test_accuracies.append(accuracy)

    print(f'\n测试集结果：平均损失：{avg_loss:.4f}, 测试准确率：{accuracy:.2f}% ({correct}/{total})')
    return accuracy


# --------------------------
# 6. 训练循环函数
# --------------------------
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler):
    """执行完整的训练循环"""
    print("\n开始训练...\n")
    best_acc = 0.0
    patience_counter = 0
    prev_lr = LEARNING_RATE

    for epoch in range(1, EPOCHS + 1):
        train(epoch, model, train_loader, criterion, optimizer)
        acc = test(model, test_loader, criterion)

        # 学习率调整
        scheduler.step(acc)

        # 使用 get_last_lr() 获取最新学习率（推荐方式）
        current_lr = scheduler.get_last_lr()[0]
        if epoch > 1 and current_lr < prev_lr:
            print(f'→ 学习率已调整：{prev_lr:.6f} -> {current_lr:.6f}')
        prev_lr = current_lr

        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_mlp_mnist.pth')
            print(f'✓ 保存最佳模型 (准确率：{best_acc:.2f}%)')
        else:
            patience_counter += 1
            print(f'⚠ 未提升，耐心计数：{patience_counter}/{PATIENCE}')

        # 早停检查
        if patience_counter >= PATIENCE:
            print(f'\n达到早停条件，停止训练')
            break

    print(f"\n训练结束！最高测试准确率：{best_acc:.2f}%")
    return best_acc


# --------------------------
# 7. 可视化函数
# --------------------------
def visualize_results(train_losses, train_accuracies, test_accuracies):
    """可视化训练结果"""
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc', marker='o')
    plt.plot(test_accuracies, label='Test Acc', marker='s')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    # 保存可视化结果
    save_dir = './results'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    print(f'\n训练曲线已保存到：{os.path.join(save_dir, "training_curves.png")}')
    plt.show()


# --------------------------
# 8. 主函数
# --------------------------
def main():
    """主函数"""
    print(f"当前使用设备：{DEVICE}")

    # 1. 创建数据加载器
    train_loader, test_loader = create_data_loaders()

    # 2. 创建模型
    model, criterion, optimizer, scheduler = create_model()

    # 3. 训练模型
    best_acc = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler)

    # 4. 可视化结果
    visualize_results(train_losses, train_accuracies, test_accuracies)


if __name__ == "__main__":
    main()
