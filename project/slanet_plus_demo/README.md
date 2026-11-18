# slanet_plus_demo

用 AntDnn 完整跑通 SLANet_plus 的 **encoder（PPLCNet + CSPPAN）+ decoder（SLAHead）** 的例子。

## 背景

SLANet_plus（百度 PP-Structure 表格结构识别模型）由三部分组成：

```
Backbone: PPLCNet   (det_pp_lcnet.py)
Neck:     CSPPAN    (csp_pan.py)
Head:     SLAHead   (table_att_head.py)  ← decoder
```

- **encoder**（PPLCNet + CSPPAN）：输入 `[488, 488, 3]` 表格图片，输出 `[256, 96]` 的
  序列特征（`fea`，供 decoder 做 attention）。
- **decoder**（SLAHead）：自回归 Attention-GRU 解码器，每一步用上一个 token 和 encoder
  特征，预测下一个 HTML 结构标签（`structure`，50 类）和单元格坐标（`loc`，8 个值）。

本 demo 用 AntDnn 的算子实现了**完整的三个模块**，并加载**真实的 SLANet_plus 权重**
（从 ONNX 转换而来）。

## 网络结构与维度

```
输入: [488, 488, 3] (NHWC)

PPLCNet (scale=1.0):
  conv1: 3x3 3->16 stride=2
  blocks2: 1 个 depthwise block 16->32
  blocks3: 2 个 depthwise block 32->64   (stride=2)   -> 输出 [122,122,64]
  blocks4: 2 个 depthwise block 64->128  (stride=2)   -> 输出 [61,61,128]
  blocks5: 6 个 depthwise block 128->256 (stride=2)   -> 输出 [31,31,256]
  blocks6: 2 个 depthwise block 256->512 (stride=2, SE) -> 输出 [16,16,512]

CSPPAN (out_channels=96):
  Channel_T: 4 尺度 1x1 统一到 96 通道
  top-down: 3 个 CSPLayer (上采样 + concat + conv)
  bottom-up: 3 个 CSPLayer (下采样 + concat + conv)
  -> 输出 4 尺度 [122,61,31,16] x 96

SLAHead (decoder):
  取最后一个尺度 [16,16,96] -> reshape [256,96] (fea)
  每步自回归 (最多 500 步):
    AttentionGRUCell:
      i2h   (linear_0): 96  -> 256   (无 bias)
      h2h   (linear_1): 256 -> 256   (有 bias)
      score (linear_2): 256 -> 1     (无 bias)
      softmax(attention) -> context
      concat(context, one_hot(token))  -> 146
      GRUCell: 146 -> 3*256  (门控)
    structure_generator: 256 -> 256 -> 50   (softmax)
    loc_generator:       256 -> 256 -> 8    (sigmoid)
    argmax 选 token; 遇 EOS(49) 停止
```

## 用到的 AntDnn 算子

| 算子 | 头文件 | 用途 |
|---|---|---|
| `Tensor` / `load_tensor_vector` | `Tensor.h` | 权重/特征容器与加载 |
| `Conv2D`（含 stride） | `Convolution.h` | PPLCNet 标准卷积 / CSPPAN 1×1 |
| `Conv2DDepthWise`（含 stride） | `ConvolutionDepthWise.h` | depthwise 卷积 |
| `BatchNorm` | `Normalize.h` | 各卷积后的 BN |
| `GlobalAvgPool2D` | `Normalize.h` | SE 模块 |
| `activation_relu/hardswish/hardsigmoid` | `Activation.h` | 激活 |
| `concat` | `Shape.h` | CSP 分叉合并、GRU 上下文拼接 |
| `UpsampleNearest2D` | `Shape.h` | top-down 上采样 |
| `one_hot` / `argmax` | `Indexing.h` | token 编码 / 自回归选 token |

> 说明：AttentionGRUCell 内部的矩阵乘（`batch_H @ i2h_w`、GRU 门控）用裸指针的
> `matvec`/`matmul` 辅助函数实现（AntDnn 风格）。SE 模块的 channel-wise 广播乘法
> 也用手写循环实现。

## 目录内容

| 文件 | 说明 |
|------|------|
| `main.cpp`        | C++ 主程序（encoder + decoder 完整前向） |
| `pplcnet.h/.cpp`  | PPLCNet backbone 实现 |
| `csppan.h/.cpp`   | CSPPAN neck 实现 |
| `export_pdparams.py` | 从 `SLANet_plus_pretrained.pdparams` 直接转换 encoder + decoder 权重（需 paddle） |
| `gen_weights.py`  | （可选）生成模拟 decoder 权重做数值比对 |
| `build_and_run.sh`| 一键转换权重 + 编译 + 运行 |

## 运行

```bash
cd project/slanet_plus_demo
./build_and_run.sh <图片路径>   # 传图片路径则读真实图片预测; 不传则用随机输入演示
```

或手动分步：

```bash
# 1. 从 .pdparams 直接转换真实权重
python3.12 export_pdparams.py   # .pdparams -> encoder.antlist + weights.antlist

# 2. 编译 (链接 OpenCV 静态库, 需 x86_64 架构)
clang++ -arch x86_64 -std=c++11 -O2 \
    -I"$ROOT/include" -I"$ROOT/include/antdnn" -I"$ROOT/3rdparty" \
    -I"$OCV/include" \
    $(find "$ROOT/modules" -name '*.cpp') \
    main.cpp pplcnet.cpp csppan.cpp \
    "$OCV/lib/libopencv_world.a" ... \
    -o slanet_plus_demo

# 3. 运行
./slanet_plus_demo test.jpg
```

权重来源：`export_pdparams.py` 默认读取
`/Users/acjog/Downloads/SLANet_plus_pretrained.pdparams`
（可用命令行参数或环境变量 `SLANET_PDPARAMS` 覆盖）。

> OpenCV 使用 `opencv-mobile` 4.14.0 framework（位于 `3rdparty/opencv-mobile/`），
> 是 fat binary（arm64 + x86_64），可 arm64 原生编译运行，无需 Rosetta。
> 注意：opencv-mobile 精简了 `imgcodecs` 模块，`imread`/`imdecode` 在
> `highgui` 模块中提供（`#include <opencv2/highgui.hpp>`）。

## 当前状态与待完成

已实现并跑通：
- ✅ PPLCNet backbone（含 SE 模块）
- ✅ CSPPAN neck（含 top-down / bottom-up）
- ✅ SLAHead decoder（AttentionGRUCell + structure/loc generator）
- ✅ 真实 encoder + decoder 权重转换（`.pdparams` → NHWC → `.antlist`，无需 ONNX）
- ✅ 真实图片读取 + 预处理（resize + normalize + pad，OpenCV）

待完成：
1. **后处理**：decoder 输出的 structure token 序列需经 TableLabelDecode 转成
   HTML 表格结构；
2. **数值验证**：与 PaddleOCR 的参考输出比对，确认前向结果一致。
