"""
作业4：SVHN模型的INT8静态量化
================================
基于作业2的SVHNNet模型，完成以下任务：
1. 手写线性量化/反量化函数
2. 模块融合 (Conv+ReLU)
3. PyTorch INT8静态量化流程 (fuse → prepare → calibrate → convert)
4. FP32 vs INT8 精度、延迟、模型大小对比
5. 每层量化误差分析 (MSE)
6. 可视化对比图表
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub, fuse_modules, prepare, convert
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from scipy.io import loadmat
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from PIL import Image

# ===================== 1. 手写线性量化函数 =====================

def linear_quantize(x, num_bits=8):
    """
    Per-Tensor 非对称线性量化 (手动实现)
    将浮点张量 x 量化为 uint8 整数张量
    公式: q = round(x / scale) + zero_point, 然后 clamp 到 [0, 2^num_bits - 1]
    """
    qmin = 0
    qmax = 2 ** num_bits - 1
    min_val = x.min().item()
    max_val = x.max().item()

    # 计算 scale 和 zero_point
    scale = (max_val - min_val) / (qmax - qmin)
    scale = max(scale, 1e-8)  # 防止除零
    zero_point = qmin - round(min_val / scale)
    zero_point = int(max(qmin, min(qmax, zero_point)))

    # 量化
    q = torch.round(x / scale) + zero_point
    q = torch.clamp(q, qmin, qmax).to(torch.uint8)
    return q, scale, zero_point


def linear_dequantize(q, scale, zero_point):
    """
    Per-Tensor 非对称线性反量化 (手动实现)
    将 uint8 整数张量反量化为浮点张量
    公式: x_hat = (q - zero_point) * scale
    """
    return (q.float() - zero_point) * scale


def test_linear_quantize():
    """测试手写量化函数的正确性"""
    print("=" * 60)
    print("手写线性量化函数测试")
    print("=" * 60)
    torch.manual_seed(42)
    x = torch.randn(4, 4)
    print(f"原始张量:\n{x}")

    q, scale, zp = linear_quantize(x)
    print(f"\n量化参数: scale={scale:.6f}, zero_point={zp}")
    print(f"量化结果 (uint8):\n{q}")

    x_deq = linear_dequantize(q, scale, zp)
    print(f"反量化结果:\n{x_deq}")

    mse = F.mse_loss(x, x_deq).item()
    print(f"\n量化误差 MSE: {mse:.6f}")
    print(f"最大绝对误差: {(x - x_deq).abs().max().item():.6f}")
    print()


# ===================== 2. SVHN数据集加载 =====================

class SVHNDataset(torch.utils.data.Dataset):
    def __init__(self, mat_file, transform=None):
        data = loadmat(mat_file)
        self.images = data['X'].transpose(3, 2, 0, 1)  # (N, C, H, W)
        self.labels = data['y'].flatten()
        self.labels[self.labels == 10] = 0
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image_pil = Image.fromarray(np.transpose(image, (1, 2, 0)))
            image = self.transform(image_pil)
        else:
            image = torch.tensor(image, dtype=torch.float32) / 255.0
        return image, label


# ===================== 3. 模型定义 =====================

class SVHNNet_FP32(nn.Module):
    """
    作业2原始模型 (不含量化桩)，用于FP32基线评估。
    结构严格匹配 homework_2/main.py 中的 SVHNNet。
    """
    def __init__(self, num_classes=10):
        super(SVHNNet_FP32, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class SVHNNet_Quantizable(nn.Module):
    """
    可量化版本的模型。
    将 nn.Sequential 拆开为单独的层，以便做模块融合 (Conv+ReLU)。
    添加 QuantStub / DeQuantStub 用于量化和反量化。
    在推理时（eval模式）Dropout不起作用，量化时Dropout会被跳过。
    """
    def __init__(self, num_classes=10):
        super(SVHNNet_Quantizable, self).__init__()
        self.quant = QuantStub()

        # 卷积块1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积块2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积块3
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.relu7 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, num_classes)

        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)

        x = self.relu1(self.conv1(x))
        x = self.pool1(self.relu2(self.conv2(x)))

        x = self.relu3(self.conv3(x))
        x = self.pool2(self.relu4(self.conv4(x)))

        x = self.relu5(self.conv5(x))
        x = self.pool3(self.relu6(self.conv6(x)))

        x = x.reshape(x.size(0), -1)
        x = self.relu7(self.fc1(x))
        x = self.fc2(x)

        x = self.dequant(x)
        return x

    def fuse_model(self):
        """模块融合：将 Conv+ReLU 融合为单个模块，提高量化精度和推理速度"""
        fuse_modules(self, [
            ['conv1', 'relu1'],
            ['conv2', 'relu2'],
            ['conv3', 'relu3'],
            ['conv4', 'relu4'],
            ['conv5', 'relu5'],
            ['conv6', 'relu6'],
            ['fc1', 'relu7'],
        ], inplace=True)


def load_weights_to_quantizable(fp32_model, quant_model):
    """
    将 FP32 模型的权重加载到可量化模型中。
    两个模型结构对应关系：
      FP32: features[0,2,4,...] / classifier[1,3]
      Quantizable: conv1,conv2,...,fc1,fc2
    """
    fp32_sd = fp32_model.state_dict()
    quant_sd = quant_model.state_dict()

    # 建立参数名映射
    mapping = {
        'features.0.weight': 'conv1.weight',   'features.0.bias': 'conv1.bias',
        'features.2.weight': 'conv2.weight',   'features.2.bias': 'conv2.bias',
        'features.6.weight': 'conv3.weight',   'features.6.bias': 'conv3.bias',
        'features.8.weight': 'conv4.weight',   'features.8.bias': 'conv4.bias',
        'features.12.weight': 'conv5.weight',  'features.12.bias': 'conv5.bias',
        'features.14.weight': 'conv6.weight',  'features.14.bias': 'conv6.bias',
        'classifier.1.weight': 'fc1.weight',   'classifier.1.bias': 'fc1.bias',
        'classifier.4.weight': 'fc2.weight',   'classifier.4.bias': 'fc2.bias',
    }

    new_sd = {}
    for fp32_name, param in fp32_sd.items():
        if fp32_name in mapping:
            new_sd[mapping[fp32_name]] = param
        else:
            # Dropout, Flatten 等无参数层跳过
            pass

    quant_model.load_state_dict(new_sd, strict=False)
    print("权重加载完成 (FP32 → Quantizable)")


# ===================== 4. 工具函数 =====================

WEIGHT_PATH = 'svhn_cnn_fp32.pth'
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'homework_2')


def get_dataloaders():
    """获取校准集和测试集 DataLoader"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = SVHNDataset(os.path.join(DATA_DIR, 'train_32x32.mat'), transform=transform)
    test_dataset = SVHNDataset(os.path.join(DATA_DIR, 'test_32x32.mat'), transform=transform)

    # 校准集: 从训练集随机抽取1000张
    np.random.seed(42)
    calib_indices = np.random.choice(len(train_dataset), 1000, replace=False)
    calib_set = Subset(train_dataset, calib_indices)

    calib_loader = DataLoader(calib_set, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    print(f"校准集大小: {len(calib_set)}")
    print(f"测试集大小: {len(test_dataset)}")
    return calib_loader, test_loader


def evaluate(model, loader, device):
    """评估模型准确率"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = correct / total
    return acc


def measure_latency(model, device, num_runs=100):
    """测量CPU单张图片推理延迟 (ms)"""
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    # 预热
    for _ in range(20):
        model(dummy_input)
    # 测量
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            model(dummy_input)
    latency = (time.perf_counter() - start) / num_runs * 1000  # ms
    return latency


def compute_layer_mse(fp32_model, int8_model, test_loader, device):
    """
    计算每层输出的量化 MSE 误差。
    对FP32模型和INT8模型注册hook，收集每层输出后计算MSE。
    """
    layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'fc1', 'fc2']
    mse_dict = {}

    fp32_outputs = {}
    int8_outputs = {}

    def get_hook(name, store):
        def hook(module, input, output):
            out = output.detach().cpu()
            # INT8模型的某些层输出是量化张量(quint8/qint8)，需要先反量化为float
            if out.dtype in (torch.quint8, torch.qint8):
                out = out.dequantize()
            store[name] = out.float() if out.is_floating_point() else out.float()
        return hook

    # 注册hook
    hooks = []
    for name in layer_names:
        fp32_layer = getattr(fp32_model, name)
        int8_layer = getattr(int8_model, name)
        hooks.append(fp32_layer.register_forward_hook(get_hook(name, fp32_outputs)))
        hooks.append(int8_layer.register_forward_hook(get_hook(name, int8_outputs)))

    fp32_model.eval()
    int8_model.eval()
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            fp32_model(imgs)
            int8_model(imgs)
            break  # 用一批数据计算MSE即可

    for name in layer_names:
        if name in fp32_outputs and name in int8_outputs:
            fp32_out = fp32_outputs[name].float()
            int8_out = int8_outputs[name]
            # 处理量化张量：反量化后转float
            if int8_out.dtype in (torch.quint8, torch.qint8):
                int8_out = int8_out.dequantize()
            int8_out = int8_out.float()
            mse = F.mse_loss(fp32_out, int8_out).item()
            mse_dict[name] = mse

    # 移除hook
    for h in hooks:
        h.remove()

    return mse_dict


def apply_manual_quantize_to_tensor(x, num_bits=8):
    """用手写的线性量化函数对张量进行量化/反量化，返回反量化后的结果"""
    q, scale, zp = linear_quantize(x, num_bits)
    return linear_dequantize(q, scale, zp)


def compare_manual_vs_pytorch_quantize(fp32_model, test_loader, device):
    """
    对比手写量化函数与 PyTorch 量化结果。
    对FP32模型的权重逐层应用手写量化，对比精度损失。
    """
    print("\n" + "=" * 60)
    print("手写量化 vs PyTorch量化 对比")
    print("=" * 60)

    # 对FP32模型的权重应用手写量化，观察每层权重的量化误差
    for name, param in fp32_model.named_parameters():
        if param.dim() >= 2:  # 只对卷积和全连接层的权重做量化
            q, scale, zp = linear_quantize(param.data)
            deq = linear_dequantize(q, scale, zp)
            mse = F.mse_loss(param.data, deq).item()
            max_err = (param.data - deq).abs().max().item()
            print(f"  {name:30s} | scale={scale:.6f} | zp={zp:4d} | MSE={mse:.8f} | MaxErr={max_err:.6f}")


# ===================== 5. 主流程 =====================

if __name__ == '__main__':
    device = torch.device('cpu')  # INT8量化仅支持CPU
    print(f"使用设备: {device}")
    print(f"PyTorch版本: {torch.__version__}\n")

    # ---- 测试手写量化函数 ----
    test_linear_quantize()

    # ---- 加载数据 ----
    calib_loader, test_loader = get_dataloaders()

    # ---- 检查权重文件 ----
    if not os.path.exists(WEIGHT_PATH):
        print(f"错误: 权重文件 '{WEIGHT_PATH}' 不存在！请先运行 train_hw2.py 训练模型。")
        exit(1)

    # ========== FP32 基线 ==========
    print("\n" + "=" * 60)
    print("FP32 基线评估")
    print("=" * 60)
    fp32_model = SVHNNet_FP32().to(device)
    fp32_model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device, weights_only=True))
    fp32_model.eval()

    fp32_acc = evaluate(fp32_model, test_loader, device)
    fp32_latency = measure_latency(fp32_model, device)
    fp32_size = os.path.getsize(WEIGHT_PATH) / (1024 ** 2)  # MB
    print(f"[FP32] 准确率: {fp32_acc*100:.2f}% | 延迟: {fp32_latency:.2f}ms | 大小: {fp32_size:.2f}MB")

    # ========== INT8 静态量化 ==========
    print("\n" + "=" * 60)
    print("INT8 静态量化")
    print("=" * 60)

    # 1) 创建可量化模型并加载权重
    int8_model = SVHNNet_Quantizable().to(device)
    load_weights_to_quantizable(fp32_model, int8_model)
    int8_model.eval()

    # 验证权重加载正确
    acc_before = evaluate(int8_model, test_loader, device)
    print(f"量化前准确率 (验证权重加载): {acc_before*100:.2f}%")

    # 2) 模块融合
    int8_model.fuse_model()
    print("模块融合完成 (Conv+ReLU)")

    # 3) 设置量化配置并准备
    int8_model.qconfig = torch.quantization.default_qconfig
    print(f"量化配置: {int8_model.qconfig}")
    prepare(int8_model, inplace=True)
    print("Prepare 完成")

    # 4) 校准
    print("开始校准 (1000张训练集样本)...")
    int8_model.eval()
    with torch.no_grad():
        for i, (imgs, _) in enumerate(calib_loader):
            int8_model(imgs.to(device))
            if (i + 1) % 5 == 0:
                print(f"  校准进度: {i+1}/{len(calib_loader)} batch")
    print("校准完成")

    # 5) 转换为INT8
    convert(int8_model, inplace=True)
    print("INT8 量化转换完成")

    # ========== INT8 评估 ==========
    int8_acc = evaluate(int8_model, test_loader, device)
    int8_latency = measure_latency(int8_model, device)
    int8_weight_path = 'svhn_cnn_int8.pth'
    torch.save(int8_model.state_dict(), int8_weight_path)
    int8_size = os.path.getsize(int8_weight_path) / (1024 ** 2)
    print(f"\n[INT8] 准确率: {int8_acc*100:.2f}% | 延迟: {int8_latency:.2f}ms | 大小: {int8_size:.2f}MB")

    # ========== 对比总结 ==========
    print("\n" + "=" * 60)
    print("FP32 vs INT8 对比总结")
    print("=" * 60)
    acc_drop = (fp32_acc - int8_acc) * 100
    speedup = fp32_latency / int8_latency if int8_latency > 0 else float('inf')
    size_ratio = fp32_size / int8_size if int8_size > 0 else float('inf')
    print(f"  准确率下降: {acc_drop:+.2f}%")
    print(f"  推理加速比: {speedup:.2f}x")
    print(f"  模型压缩比: {size_ratio:.2f}x")

    # ========== 手写量化 vs PyTorch量化对比 ==========
    compare_manual_vs_pytorch_quantize(fp32_model, test_loader, device)

    # ========== 每层量化误差分析 ==========
    print("\n" + "=" * 60)
    print("每层量化误差分析 (MSE)")
    print("=" * 60)

    # 构建一个融合后的FP32模型用于对比（结构与int8_model对应）
    fp32_fused = SVHNNet_Quantizable().to(device)
    load_weights_to_quantizable(fp32_model, fp32_fused)
    fp32_fused.eval()
    fp32_fused.fuse_model()

    mse_dict = compute_layer_mse(fp32_fused, int8_model, test_loader, device)
    for name, mse in mse_dict.items():
        print(f"  {name:10s} MSE: {mse:.8f}")

    # ========== 可视化 ==========
    print("\n生成可视化图表...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) 精度对比
    ax = axes[0, 0]
    bars = ax.bar(['FP32', 'INT8'], [fp32_acc * 100, int8_acc * 100],
                  color=['steelblue', 'coral'], width=0.5)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Accuracy Comparison')
    for bar, val in zip(bars, [fp32_acc * 100, int8_acc * 100]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.2f}%', ha='center', fontsize=11)

    # 2) 延迟对比
    ax = axes[0, 1]
    bars = ax.bar(['FP32', 'INT8'], [fp32_latency, int8_latency],
                  color=['steelblue', 'coral'], width=0.5)
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Inference Latency (CPU, single image)')
    for bar, val in zip(bars, [fp32_latency, int8_latency]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.2f}ms', ha='center', fontsize=11)

    # 3) 模型大小对比
    ax = axes[1, 0]
    bars = ax.bar(['FP32', 'INT8'], [fp32_size, int8_size],
                  color=['steelblue', 'coral'], width=0.5)
    ax.set_ylabel('Model Size (MB)')
    ax.set_title('Model Size Comparison')
    for bar, val in zip(bars, [fp32_size, int8_size]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.2f}MB', ha='center', fontsize=11)

    # 4) 各层MSE
    ax = axes[1, 1]
    names = list(mse_dict.keys())
    values = list(mse_dict.values())
    bars = ax.bar(names, values, color='mediumpurple')
    ax.set_ylabel('MSE')
    ax.set_title('Quantization MSE per Layer')
    ax.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.2e}', ha='center', fontsize=8, rotation=30)

    plt.tight_layout()
    plt.savefig('quantization_results.png', dpi=150, bbox_inches='tight')
    print("图表已保存为 quantization_results.png")

    # ========== 汇总 ==========
    print("\n" + "=" * 60)
    print("作业4完成！")
    print("=" * 60)
    print(f"FP32模型: 准确率={fp32_acc*100:.2f}%, 延迟={fp32_latency:.2f}ms, 大小={fp32_size:.2f}MB")
    print(f"INT8模型: 准确率={int8_acc*100:.2f}%, 延迟={int8_latency:.2f}ms, 大小={int8_size:.2f}MB")
    print(f"准确率下降: {acc_drop:+.2f}%, 加速比: {speedup:.2f}x, 压缩比: {size_ratio:.2f}x")
