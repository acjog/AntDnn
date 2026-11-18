# -*- coding: utf-8 -*-
"""
生成 cnn_demo 所需的测试权重文件。

用法:  python3 gen_weights.py

生成的文件:
  - weights.antlist : 训练好的权重(模拟 keras 导出的权重列表)
  - img.antts       : 测试输入图像(1x8x8x1 单通道灰度图)

权重 layout 与 antdnn 保持一致:
  - Conv2D 权重: [kh, kw, cin, cout]
  - Conv2D bias : [cout]
  - Dense  权重: [in, out]
  - Dense  bias : [out]
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from antdnn import np_to_file, nplist_to_file


def main():
    # ---------- 网络结构设计 ----------
    # 输入: 1 x 8 x 8 x 1 (NHWC, 单 batch, 单通道灰度图)
    # 网络:
    #   Conv2D(3x3, 1 -> 4, SAME) + ReLU
    #   MaxPooling2D(2x2)          -> 4x4x4
    #   Conv2D(3x3, 4 -> 8, VALID) + ReLU
    #   MaxPooling2D(2x2)          -> 1x1x8
    #   Flatten                    -> 8
    #   Dense(8 -> 3) + Softmax    -> 3 分类
    #
    # 权重列表顺序(与 main.cpp 中的 tsvec 索引一一对应):
    #   [0] conv1_w, [1] conv1_b,
    #   [2] conv2_w, [3] conv2_b,
    #   [4] fc_w,    [5] fc_b

    rng = np.random.RandomState(42)  # 固定种子, 结果可复现

    # conv1: 3x3x1x4
    conv1_w = (rng.rand(3, 3, 1, 4) - 0.5).astype(np.float32)
    conv1_b = (rng.rand(4) - 0.5).astype(np.float32)

    # conv2: 3x3x4x8
    conv2_w = (rng.rand(3, 3, 4, 8) - 0.5).astype(np.float32)
    conv2_b = (rng.rand(8) - 0.5).astype(np.float32)

    # fc: 8 -> 3
    fc_w = (rng.rand(8, 3) - 0.5).astype(np.float32)
    fc_b = (rng.rand(3) - 0.5).astype(np.float32)

    weights = [conv1_w, conv1_b,
               conv2_w, conv2_b,
               fc_w, fc_b]

    # 测试输入图像: 8x8x1 (注意: antdnn 的输入是 3 维 [H, W, C], 无 batch 维度)
    img = (rng.rand(8, 8, 1) * 255).astype(np.float32)

    # 写出文件
    here = os.path.dirname(os.path.abspath(__file__))
    nplist_to_file(weights, os.path.join(here, "weights.antlist"))
    np_to_file(img, os.path.join(here, "img.antts"))

    print("生成完成:")
    print("  weights.antlist : %d 个权重张量" % len(weights))
    print("  img.antts       : 输入图像 %s" % str(img.shape))

    # ---------- 用 numpy 直接算一遍前向, 作为参考输出 ----------
    ref = reference_forward(img, conv1_w, conv1_b, conv2_w, conv2_b, fc_w, fc_b)
    print("\n参考输出 (numpy 前向计算结果, 供比对):")
    print(ref)
    np.savetxt(os.path.join(here, "ref_output.txt"), ref.reshape(1, -1), fmt="%.6f")


def relu(x):
    return np.maximum(x, 0.0)


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def conv2d(x, w, b, same=True):
    """朴素卷积实现, 用于生成参考输出。x: [H,W,C], w: [kh,kw,cin,cout]"""
    H, W, C = x.shape[0], x.shape[1], x.shape[2]
    kh, kw, cin, cout = w.shape
    assert cin == C, "通道数不匹配"
    if same:
        pad_h, pad_w = (kh - 1) // 2, (kw - 1) // 2
        xp = np.pad(x, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))
        out_h, out_w = H, W
    else:
        xp = x
        out_h, out_w = H - kh + 1, W - kw + 1
    out = np.zeros((out_h, out_w, cout), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            patch = xp[i:i + kh, j:j + kw, :]  # [kh,kw,cin]
            for c in range(cout):
                out[i, j, c] = np.sum(patch * w[:, :, :, c]) + b[c]
    return out


def maxpool2d(x, p):
    H, W, C = x.shape[0], x.shape[1], x.shape[2]
    out_h, out_w = H // p, W // p
    out = np.zeros((out_h, out_w, C), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            out[i, j, :] = np.max(
                x[i * p:(i + 1) * p, j * p:(j + 1) * p, :], axis=(0, 1))
    return out


def reference_forward(img, conv1_w, conv1_b, conv2_w, conv2_b, fc_w, fc_b):
    x = img
    x = relu(conv2d(x, conv1_w, conv1_b, same=True))   # 8x8x4
    x = maxpool2d(x, 2)                                  # 4x4x4
    x = relu(conv2d(x, conv2_w, conv2_b, same=False))   # 2x2x8
    x = maxpool2d(x, 2)                                  # 1x1x8
    x = x.reshape(1, -1)                                 # 8
    x = x @ fc_w + fc_b                                  # 3
    x = softmax(x)
    return x


if __name__ == "__main__":
    main()
