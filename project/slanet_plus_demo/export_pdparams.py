# -*- coding: utf-8 -*-
"""
从 SLANet_plus_pretrained.pdparams 直接转换 encoder + decoder 权重为 AntDnn .antlist,
彻底摆脱 ONNX。

用法: python3.12 export_pdparams.py [SLANet_plus_pretrained.pdparams]

输出:
  encoder.antlist  (PPLCNet + CSPPAN 权重, NHWC)
  weights.antlist  (SLAHead decoder 权重)

依赖: paddle (仅用于 paddle.load 读取 .pdparams)

权重转换规则:
  - 卷积权重 NCHW [cout, cin/groups, kh, kw] -> NHWC [kh, kw, cin/groups, cout]
  - depthwise 权重 [C, 1, kh, kw] -> [kh, kw, 1, C]
  - BN: pdparams 存 weight/bias/_mean/_variance
         AntDnn BatchNorm 需要 (mean, var, scale, bias)
         即: _mean->mean, _variance->var, weight->scale, bias->bias
  - GRU: pdparams 存 weight_ih [3H, I] / weight_hh [3H, H] (Paddle 布局)
         AntDnn 需要 [I, 3H] / [H, 3H], 需转置
"""

import numpy as np
import os
import sys
import paddle


def _as_np(arr):
    """把 paddle.Tensor / numpy array 统一转成 numpy float32 array"""
    if hasattr(arr, "numpy"):  # paddle.Tensor
        arr = arr.numpy()
    return np.asarray(arr, dtype=np.float32)


def nplist_to_file(nplist, filename):
    """内联实现 antdnn 的 .antlist 写出"""
    with open(filename, "wb") as f:
        f.write(b"ant\0")
        f.write(b"lis\0")
        f.write(np.array([len(nplist)], dtype=np.int32).tobytes())
        for arr in nplist:
            arr = _as_np(arr)
            f.write(b"tsd\0")
            header = [len(arr.shape)] + list(arr.shape)
            f.write(np.array(header, dtype=np.int32).tobytes())
            f.write(arr.tobytes())


def nchw_to_nhwc(w):
    """[cout, cin/g, kh, kw] -> [kh, kw, cin/g, cout]"""
    return np.transpose(w, (2, 3, 1, 0))


def bn_4param(state, prefix):
    """读取 BN 四参数, 返回 (mean, var, scale, bias), 均为 [C]"""
    mean = state[prefix + "._mean"]
    var = state[prefix + "._variance"]
    scale = state[prefix + ".weight"]
    bias = state[prefix + ".bias"]
    return mean, var, scale, bias


class Exporter:
    def __init__(self, state):
        self.state = state
        self.encoder = []   # 按 C++ 消费顺序
        self.decoder = []

    def conv_bn(self, prefix):
        """ConvBNLayer: conv_w + bn4"""
        w = nchw_to_nhwc(self.state[prefix + ".conv.weight"])
        mean, var, scale, bias = bn_4param(self.state, prefix + ".bn")
        self.encoder += [w, mean, var, scale, bias]

    def depthwise_separable(self, prefix, use_se):
        """DepthwiseSeparable: dw + [se] + pw"""
        # dw_conv
        dw_w = nchw_to_nhwc(self.state[prefix + ".dw_conv.conv.weight"])
        mean, var, scale, bias = bn_4param(self.state, prefix + ".dw_conv.bn")
        self.encoder += [dw_w, mean, var, scale, bias]
        # se (可选): fc1(w+b) + fc2(w+b)
        if use_se:
            se_fc1_w = nchw_to_nhwc(self.state[prefix + ".se.conv1.weight"])
            se_fc1_b = self.state[prefix + ".se.conv1.bias"]
            se_fc2_w = nchw_to_nhwc(self.state[prefix + ".se.conv2.weight"])
            se_fc2_b = self.state[prefix + ".se.conv2.bias"]
            self.encoder += [se_fc1_w, se_fc1_b, se_fc2_w, se_fc2_b]
        # pw_conv
        pw_w = nchw_to_nhwc(self.state[prefix + ".pw_conv.conv.weight"])
        mean, var, scale, bias = bn_4param(self.state, prefix + ".pw_conv.bn")
        self.encoder += [pw_w, mean, var, scale, bias]

    def dp_module(self, prefix):
        """DPModule (CSPPAN): dwconv(bn1) + pwconv(bn2)"""
        dw_w = nchw_to_nhwc(self.state[prefix + ".dwconv.weight"])
        mean, var, scale, bias = bn_4param(self.state, prefix + ".bn1")
        self.encoder += [dw_w, mean, var, scale, bias]
        pw_w = nchw_to_nhwc(self.state[prefix + ".pwconv.weight"])
        mean, var, scale, bias = bn_4param(self.state, prefix + ".bn2")
        self.encoder += [pw_w, mean, var, scale, bias]

    def csp_layer(self, prefix):
        """CSPLayer: short_conv + main_conv + block.conv1 + block.conv2 + final_conv
        顺序需与 csppan.cpp 的 csp_layer 一致:
          short_conv -> main_conv -> block.conv1 -> block.conv2(dp) -> final_conv
        """
        # short_conv
        self.conv_bn(prefix + ".short_conv")
        # main_conv
        self.conv_bn(prefix + ".main_conv")
        # block.0.conv1 (1x1)
        self.conv_bn(prefix + ".blocks.0.conv1")
        # block.0.conv2 (DPModule 5x5)
        self.dp_module(prefix + ".blocks.0.conv2")
        # final_conv
        self.conv_bn(prefix + ".final_conv")

    def export_backbone(self):
        s = self.state
        # conv1: 3x3 3->16 stride=2
        self.conv_bn("backbone.conv1")
        # blocks2: 1 block
        self.depthwise_separable("backbone.blocks2.0", use_se=False)
        # blocks3: 2 blocks
        self.depthwise_separable("backbone.blocks3.0", use_se=False)
        self.depthwise_separable("backbone.blocks3.1", use_se=False)
        # blocks4: 2 blocks
        self.depthwise_separable("backbone.blocks4.0", use_se=False)
        self.depthwise_separable("backbone.blocks4.1", use_se=False)
        # blocks5: 6 blocks
        for i in range(6):
            self.depthwise_separable(f"backbone.blocks5.{i}", use_se=False)
        # blocks6: 2 blocks (use_se=True)
        self.depthwise_separable("backbone.blocks6.0", use_se=True)
        self.depthwise_separable("backbone.blocks6.1", use_se=True)

    def export_neck(self):
        # conv_t: 4 个 1x1
        for i in range(4):
            self.conv_bn(f"neck.conv_t.convs.{i}")
        # top-down: 3 个 CSPLayer (顺序 idx=2,1,0 对应 forward 的 N-1-idx)
        # 实际 forward 按 idx=3,2,1 遍历, 对应 top_down_blocks[0], [1], [2]
        for i in range(3):
            self.csp_layer(f"neck.top_down_blocks.{i}")
        # bottom-up: 3 个 (downsample + csp_layer)
        for i in range(3):
            self.dp_module(f"neck.downsamples.{i}")
            self.csp_layer(f"neck.bottom_up_blocks.{i}")

    def export_head(self):
        s = self.state
        h = "head.structure_attention_cell"
        # i2h (无 bias)
        self.decoder.append(s[h + ".i2h.weight"])
        # h2h (有 bias)
        self.decoder.append(s[h + ".h2h.weight"])
        self.decoder.append(s[h + ".h2h.bias"])
        # score (无 bias)
        self.decoder.append(s[h + ".score.weight"])
        # GRU: weight_ih [3H, I] -> [I, 3H], weight_hh [3H, H] -> [H, 3H]
        self.decoder.append(np.transpose(s[h + ".rnn.weight_ih"], (1, 0)))  # gru_wx [146,768]
        self.decoder.append(np.transpose(s[h + ".rnn.weight_hh"], (1, 0)))  # gru_wh [256,768]
        self.decoder.append(s[h + ".rnn.bias_ih"])                           # gru_bx [768]
        self.decoder.append(s[h + ".rnn.bias_hh"])                           # gru_bh [768]
        # structure_generator: 0(256->256) + 1(256->50)
        self.decoder.append(s["head.structure_generator.0.weight"])
        self.decoder.append(s["head.structure_generator.0.bias"])
        self.decoder.append(s["head.structure_generator.1.weight"])
        self.decoder.append(s["head.structure_generator.1.bias"])
        # loc_generator: 0(256->256) + 1(256->8)
        self.decoder.append(s["head.loc_generator.0.weight"])
        self.decoder.append(s["head.loc_generator.0.bias"])
        self.decoder.append(s["head.loc_generator.1.weight"])
        self.decoder.append(s["head.loc_generator.1.bias"])


def main():
    if len(sys.argv) > 1:
        pdparams_path = sys.argv[1]
    else:
        pdparams_path = os.environ.get(
            "SLANET_PDPARAMS",
            "/Users/acjog/Downloads/SLANet_plus_pretrained.pdparams",
        )

    print(f"加载 {pdparams_path} ...")
    state = paddle.load(pdparams_path)
    print(f"共 {len(state)} 个权重张量")

    exp = Exporter(state)
    exp.export_backbone()
    print(f"backbone 导出 {len(exp.encoder)} 个张量")
    exp.export_neck()
    print(f"backbone+neck 导出 {len(exp.encoder)} 个张量")
    exp.export_head()
    print(f"decoder 导出 {len(exp.decoder)} 个张量")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    nplist_to_file(exp.encoder, os.path.join(out_dir, "encoder.antlist"))
    nplist_to_file(exp.decoder, os.path.join(out_dir, "weights.antlist"))
    print(f"\n已生成:")
    print(f"  encoder.antlist: {len(exp.encoder)} 个张量")
    print(f"  weights.antlist: {len(exp.decoder)} 个张量")


if __name__ == "__main__":
    main()
