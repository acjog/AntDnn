# -*- coding: utf-8 -*-
"""
从 encoder.onnx 提取 PPLCNet + CSPPAN 的权重, 转换为 AntDnn 的 .antlist 格式 (NHWC)。

权重转换: ONNX 权重是 NCHW [cout, cin/groups, kh, kw]
         AntDnn 需要 NHWC [kh, kw, cin/groups, cout]

导出顺序严格按照源码 forward 的执行顺序:
  PPLCNet: conv1 -> blocks2 -> blocks3 -> blocks4 -> blocks5 -> blocks6
  CSPPAN:  conv_t(4个1x1) -> top-down(3个CSPLayer) -> bottom-up(3个downsample+3个CSPLayer)

每个 ConvBNLayer 导出 3 个张量: [conv_w(NHWC), bn_weight, bn_bias]
每个 DPModule(depthwise+pointwise) 导出 6 个张量: dw_w, dw_bn_w, dw_bn_b, pw_w, pw_bn_w, pw_bn_b
SE 模块(blocks6 的 use_se=True)额外导出: se_fc1_w, se_fc2_w (1x1, 无 BN 无 bias)

用法: python3 export_encoder.py
输出: encoder.antlist  (供 main.cpp 的 load_tensor_vector 加载)
"""

import onnx
import numpy as np
import os
import sys

# encoder.onnx 路径: 可用命令行参数或环境变量指定
if len(sys.argv) > 1:
    ONNX_PATH = sys.argv[1]
else:
    ONNX_PATH = os.environ.get(
        'SLANET_ENCODER_ONNX',
        '/Users/acjog/Documents/code/ncnn-webassembly-ppocr/SLANet_plus/encoder.onnx'
    )

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(OUT_DIR, 'encoder.antlist')

m = onnx.load(ONNX_PATH)
inits = {i.name: i for i in m.graph.initializer}

# 按拓扑序收集所有 Conv / BatchNormalization 节点
conv_nodes = [n for n in m.graph.node if n.op_type == 'Conv']
bn_nodes = [n for n in m.graph.node if n.op_type == 'BatchNormalization']

# 建立 conv 输出名 -> conv 节点映射 (便于按执行顺序取权重)
conv_by_out = {}
for n in conv_nodes:
    conv_by_out[n.output[0]] = n

def get_arr(name):
    return onnx.numpy_helper.to_array(inits[name])

def nchw_to_nhwc(w):
    """[cout, cin/g, kh, kw] -> [kh, kw, cin/g, cout]"""
    return np.transpose(w, (2, 3, 1, 0))

# 按 conv 节点顺序导出 ConvBNLayer 的权重 (conv_w + bn_w + bn_b)
def export_convbn(conv_node):
    """返回 [conv_w_nhwc, bn_w, bn_b, bn_mean, bn_var]"""
    w = nchw_to_nhwc(get_arr(conv_node.input[1]))
    # 找到该 conv 输出对应的 BN
    bn = None
    for b in bn_nodes:
        if b.input[0] == conv_node.output[0]:
            bn = b
            break
    if bn is None:
        raise RuntimeError(f'conv {conv_node.output[0]} 没有对应的 BN')
    bn_w = get_arr(bn.input[1])   # scale
    bn_b = get_arr(bn.input[2])   # bias
    bn_mean = get_arr(bn.input[3])
    bn_var = get_arr(bn.input[4])
    return w, bn_w, bn_b, bn_mean, bn_var

# 所有要导出的权重列表 (每个元素是 numpy array)
weights = []

# ============ PPLCNet backbone ============
# 通过输出名按执行顺序定位每个 Conv

def find_conv(name):
    for n in conv_nodes:
        if n.output[0] == name:
            return n
    raise RuntimeError(f'找不到 conv: {name}')

# conv1: 3x3, 3->16, stride=2
def export_layer(conv_name):
    w, bn_w, bn_b, bn_mean, bn_var = export_convbn(find_conv(conv_name))
    # AntDnn BatchNorm 需要 [mean, var, scale, bias]
    weights.append(w)
    weights.append(bn_mean)
    weights.append(bn_var)
    weights.append(bn_w)
    weights.append(bn_b)

# 由于 layer 名称有规律, 按序号导出。先导出 backbone 的 conv+depthwise 序列
# 从前面分析: conv2d.N 是 1x1, depthwise_conv2d.N 是 depthwise
# 顺序: conv1(conv2d.0), blocks2(dw.0 + conv2d.1), blocks3(dw.1+conv2d.2, dw.2+conv2d.3), ...

# 为了稳妥, 直接按拓扑序导出所有 Conv 及其 BN (拓扑序 == 执行序, 已验证)
# 但要区分: 哪些是 ConvBNLayer(普通conv+dw), 哪些是 SE 的裸 1x1 conv
# SE 的 conv 无 BN (conv2d.12/13, conv2d.15/16), 需要特殊处理

# 策略: 遍历拓扑序, 遇到 Conv 就判断是否有对应 BN
# 有 BN -> ConvBNLayer (导出 conv_w + bn 4参数)
# 无 BN -> SE 的 1x1 (导出 conv_w 即可, 但 SE 的 conv 也有 bias? 看源码 SEModule 无 bias)

print('开始导出权重...')

se_convs = []  # 记录 SE 的 conv (无 BN)
for n in conv_nodes:
    # 判断是否有 BN 紧跟
    has_bn = any(b.input[0] == n.output[0] for b in bn_nodes)
    if has_bn:
        w, bn_w, bn_b, bn_mean, bn_var = export_convbn(n)
        weights.append(w)
        weights.append(bn_mean)
        weights.append(bn_var)
        weights.append(bn_w)
        weights.append(bn_b)
    else:
        # SE 模块的 1x1 conv, 无 BN 无 bias (源码 SEModule 的 conv1/conv2 无 bias)
        w = nchw_to_nhwc(get_arr(n.input[1]))
        weights.append(w)

print(f'共导出 {len(weights)} 个权重张量')

# 内联实现 antdnn 的 .antlist 文件写出 (避免路径耦合)
def np_to_file_internal(npdata, f):
    f.write('tsd'.encode())
    f.write(b'\0')
    header = [len(npdata.shape)] + list(npdata.shape)
    f.write(np.array(header, dtype=np.int32).tobytes())
    f.write(npdata.astype(np.float32).tobytes())

def nplist_to_file_internal(nplist, filename):
    with open(filename, 'wb') as f:
        f.write('ant'.encode())
        f.write(b'\0')
        f.write('lis'.encode())
        f.write(b'\0')
        header = [len(nplist)]
        f.write(np.array(header, dtype=np.int32).tobytes())
        for npdata in nplist:
            np_to_file_internal(npdata, f)

nplist_to_file_internal(weights, OUT_FILE)
print(f'saved {OUT_FILE} ({len(weights)} tensors)')

# 打印每个权重 shape, 供 main.cpp 对照
for i, w in enumerate(weights):
    print(f'  [{i:3d}] {str(w.shape)}')
