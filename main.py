# -*- coding: utf-8 -*-
# python main.py --image test1.jpg
import time
import argparse
import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms

print(f"torch版本 {torch.__version__}, torchvision版本 {torchvision.__version__}")

from utils import label_to_onehot, cross_entropy_for_onehot

time1 = time.time()

parser = argparse.ArgumentParser(description="Deep Leakage from Gradients.")
parser.add_argument(
    "--index", type=int, default="25", help="the index for leaking images on CIFAR."
)
parser.add_argument(
    "--image", type=str, default="", help="the path to customized image."
)
args = parser.parse_args()

device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda"
print("Running on %s" % device)

dst = datasets.CIFAR100("~/.torch", download=True)  # 下载并加载CIFAR100数据集
tp = transforms.ToTensor()  # 将图像转换为张量（tensor）格式
tt = transforms.ToPILImage()  # 将张量格式的图像转换为PIL图像格式

# 获取图片数据
img_index = args.index
gt_data = tp(dst[img_index][0]).to(device)  # 模式1：加载CIFAR数据集的图片
if len(args.image) > 1:
    gt_data = Image.open(args.image)  # 模式2：加载自定义图片，
    gt_data = tp(gt_data).to(device)  # 转换为张量。


gt_data = gt_data.view(1, *gt_data.size())  # 将原始图像数据的形状调整为(1, 通道数, Hight, Width)
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)  # 创建了一个包含图像标签的张量
gt_label = gt_label.view(  # 将图像标签的形状调整为(1,)
    1,
)
gt_onehot_label = label_to_onehot(gt_label)  # 将图像标签转换为独热编码格式

plt.imshow(tt(gt_data[0].cpu()))  # 将处理后的图像数据可视化

from models.vision import LeNet, weights_init

# net = LeNet().to(device) ### 创建网络
net = LeNet()  ### 在CPU创建网络


torch.manual_seed(1234)  # 设置随机种子

net.apply(weights_init)  # 初始化权重
criterion = cross_entropy_for_onehot  # 损失函数定为交叉熵误差

net = net.to(device)  ### 将网络转移到指定设备

# compute original gradient
# 计算原始梯度（真实梯度）
pred = net(gt_data)  # 一次前向传播
# 由于是模拟DLG的过程，所以真实学习的过程不重要，gt_onehot_label无需真实
y = criterion(pred, gt_onehot_label)  # 求损失值
dy_dx = torch.autograd.grad(y, net.parameters())  # 损失函数对模型参数的梯度dy_dx

# 创建一个列表，并将梯度 去除追踪历史后 填入列表。
original_dy_dx = list((_.detach().clone() for _ in dy_dx))
print(f"相等：{torch.all(dy_dx[0] == original_dy_dx[0])}")

# generate dummy data and label
# 随机生成伪数据和标签
# 创建两个张量 dummy_data 和 dummy_label ，它们的形状与 gt_data 和 gt_onehot_label 相同。
# 初始值从标准正态分布中随机采样得到，requires_grad_(True)是为了在后续的优化过程中跟踪它们的梯度。
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)  # 图片数据
dummy_label = (  # 图片标签
    torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
)

plt.imshow(tt(dummy_data[0].cpu()))

# 创建一个LBFGS优化器对象
# LBFGS是一种基于拟牛顿法的优化算法，用于最小化给定的损失函数。
optimizer = torch.optim.LBFGS([dummy_data, dummy_label])


time2 = time.time()
history = []  # 保存迭代中的历史记录
for iters in range(300):
    dummy_dy_dx_ = None  ### 保存本轮的伪梯度

    # 定义一个闭包函数，返回伪梯度与起源真梯度之间的差异
    def closure():
        global dummy_dy_dx_
        # 0. 将优化器的梯度缓存清零。
        optimizer.zero_grad()
        # 1. 用伪数据进行一次前向传播，得到伪结果。
        dummy_pred = net(dummy_data)
        # 2. 用交叉熵误差求伪结果对伪标签的伪损失值。
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label)
        # 3. 计算伪损失值对神经网络伪参数的梯度。
        dummy_dy_dx = torch.autograd.grad(  # create_graph 创建计算图，以支持高阶求导
            dummy_loss, net.parameters(), create_graph=True
        )
        dummy_dy_dx_ = dummy_dy_dx  ### 保存本轮的伪梯度。
        # 4. 计算当前迭代中，伪梯度 dummy_dy_dx 与真实梯度 original_dy_dx 之间的差异
        grad_diff = 0  # grad_diff 初始为int，后面变为单个元素张量
        #    遍历当前伪梯度和起源梯度，计算它们之间每项平方差的总和
        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
            grad_diff += ((gx - gy) ** 2).sum()
        # 5. 反向传播，计算张量 grad_diff 的梯度，存储在它的 grad 属性中。
        grad_diff.backward()
        return grad_diff

    # 执行一次优化，更新 dummy_data 和 dummy_label 的值
    optimizer.step(closure)

    if iters % 10 == 0:
        history.append(tt(dummy_data[0].cpu()))
        current_loss = closure()
        print(iters, "%.4f" % current_loss.item())
        # print(f"梯度：{dummy_dy_dx_[0][0][0][0]}")  ### 输出本轮的伪梯度


time3 = time.time()
print(f"\n{device} 总耗时 {time3-time1} ，DLG耗时 {time3-time2}")

plt.figure(figsize=(12, 8))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(history[i])
    plt.title("iter=%d" % (i * 10))
    plt.axis("off")

plt.show()
