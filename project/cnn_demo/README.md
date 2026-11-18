# cnn_demo

一个能直接跑通的完整 CNN 前向推理 demo，演示 AntDnn 的完整用法：
从 keras 风格导出的权重文件加载参数，手动搭建网络结构，完成前向计算，
并与 numpy 独立计算的参考结果做数值比对。

## 网络结构

```
输入      1x8x8x1  (NHWC, 单通道灰度图)
Conv2D    3x3, 1 -> 4, PADDING_SAME  + ReLU
MaxPool   2x2                        -> 4x4x4
Conv2D    3x3, 4 -> 8, PADDING_VALID + ReLU
MaxPool   2x2                        -> 1x1x8
Flatten                              -> 8
Dense     8 -> 3
Softmax                              -> 3 类概率
```

## 目录内容

| 文件 | 说明 |
|------|------|
| `gen_weights.py`  | 用 numpy 生成测试权重(`weights.antlist`)和输入(`img.antts`)，并算出参考输出(`ref_output.txt`) |
| `main.cpp`        | C++ 推理主程序 |
| `build_and_run.sh`| 一键编译运行脚本 |
| `antdnn.py`       | 软链接到 `python/antdnn.py`，提供权重文件读写函数 |

## 运行

```bash
cd project/cnn_demo
./build_and_run.sh
```

或手动分步：

```bash
# 1. 生成权重
python3 gen_weights.py

# 2. 编译 (ROOT 为项目根目录)
clang++ -std=c++11 -O2 \
    -I"$ROOT/include" -I"$ROOT/include/antdnn" -I"$ROOT/3rdparty" \
    "$ROOT/modules/"*.cpp main.cpp -o cnn_demo

# 3. 运行
./cnn_demo
```

## 说明

- `weights.antlist` 中权重顺序与 `main.cpp` 的 `tsvec` 索引一一对应：
  `[0] conv1_w, [1] conv1_b, [2] conv2_w, [3] conv2_b, [4] fc_w, [5] fc_b`
- 权重 layout 与 antdnn 约定一致：
  - Conv2D 权重 `[kh, kw, cin, cout]`，bias `[cout]`
  - Dense 权重 `[in, out]`，bias `[out]`
- 程序会打印最终 softmax 概率、预测类别，并与 numpy 参考输出比对
  （最大误差 < 1e-3 即判定通过）。
