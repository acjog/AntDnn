# -*- coding: utf-8 -*-
"""
生成 slanet_plus_demo 所需的模拟权重 + 参考输出。

SLANet_plus 的 decoder 是一个「自回归 Attention-GRU 解码器」, 结构与
ppocr/modeling/heads/table_att_head.py 的 SLAHead 一致。本脚本:
  1. 生成 decoder 各层的模拟权重 (固定随机种子, 可复现);
  2. 用 numpy 直接实现一遍 AttentionGRUCell + structure/loc generator 的前向,
     作为 C++ 端的参考输出;
  3. 写出权重文件 + 参考输出文件, 供 main.cpp 加载比对。

权重 layout 与 antdnn 约定一致 (Dense 权重为 [in, out], bias 为 [out])。

维度 (与 SLANet_plus.yml / table_structure.cpp 一致):
  ENC_SEQ = 64     # encoder 输出序列长度 (为演示方便取 64, 真实为 256)
  ENC_FEAT = 96    # encoder 输出特征维
  DEC_HID  = 256   # GRU hidden size
  NUM_CLS  = 30    # structure 字典大小 (table_structure_dict)
  LOC_NUM  = 8     # 坐标回归数 (loc_reg_num)
  MAX_STEPS = 500  # 最大解码步数
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
from antdnn import np_to_file, nplist_to_file


# ---- 维度常量 (与真实 SLANet_plus 一致) ----
ENC_SEQ = 256
ENC_FEAT = 96
DEC_HID = 256
NUM_CLS = 50
LOC_NUM = 8
GRU_IN = ENC_FEAT + NUM_CLS  # 96 + 50 = 146 (GRU 输入 = context + one-hot)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def make_weights():
    rng = np.random.RandomState(42)

    w = {}

    # AttentionGRUCell 的三个 Linear:
    #   i2h  (linear_0): ENC_FEAT -> DEC_HID   [96, 256]   (无 bias)
    #   h2h  (linear_1): DEC_HID  -> DEC_HID   [256, 256]  (有 bias)
    #   score(linear_2): DEC_HID  -> 1         [256, 1]    (无 bias)
    w["i2h_w"] = (rng.randn(ENC_FEAT, DEC_HID) * 0.1).astype(np.float32)
    w["h2h_w"] = (rng.randn(DEC_HID, DEC_HID) * 0.1).astype(np.float32)
    w["h2h_b"] = (rng.randn(DEC_HID) * 0.1).astype(np.float32)
    w["score_w"] = (rng.randn(DEC_HID, 1) * 0.1).astype(np.float32)

    # GRUCell: weight_ih [GRU_IN, 3*DEC_HID], weight_hh [DEC_HID, 3*DEC_HID]
    #          bias_ih  [3*DEC_HID], bias_hh [3*DEC_HID]
    # 注意: paddle GRUCell 的 weight_ih 是 (3*hidden, input), 这里转成
    #       antdnn 的 Dense 布局 [in, out] = [input, 3*hidden]
    w["gru_wx"] = (rng.randn(GRU_IN, 3 * DEC_HID) * 0.1).astype(np.float32)
    w["gru_wh"] = (rng.randn(DEC_HID, 3 * DEC_HID) * 0.1).astype(np.float32)
    w["gru_bx"] = (rng.randn(3 * DEC_HID) * 0.1).astype(np.float32)
    w["gru_bh"] = (rng.randn(3 * DEC_HID) * 0.1).astype(np.float32)

    # structure_generator: hidden -> hidden -> NUM_CLS  (两个 Linear)
    w["fc1_w"] = (rng.randn(DEC_HID, DEC_HID) * 0.1).astype(np.float32)
    w["fc1_b"] = (rng.randn(DEC_HID) * 0.1).astype(np.float32)
    w["struct_w"] = (rng.randn(DEC_HID, NUM_CLS) * 0.1).astype(np.float32)
    w["struct_b"] = (rng.randn(NUM_CLS) * 0.1).astype(np.float32)

    # loc_generator: hidden -> hidden -> LOC_NUM -> sigmoid
    w["fc2_w"] = (rng.randn(DEC_HID, DEC_HID) * 0.1).astype(np.float32)
    w["fc2_b"] = (rng.randn(DEC_HID) * 0.1).astype(np.float32)
    w["loc_w"] = (rng.randn(DEC_HID, LOC_NUM) * 0.1).astype(np.float32)
    w["loc_b"] = (rng.randn(LOC_NUM) * 0.1).astype(np.float32)

    return w


def attention_gru_cell(prev_hidden, batch_H, char_onehot, w):
    """
    与 AttentionGRUCell.forward 一一对应的 numpy 实现。
    prev_hidden: [DEC_HID]
    batch_H:     [ENC_SEQ, ENC_FEAT]
    char_onehot: [NUM_CLS]
    """
    # i2h: batch_H @ i2h_w  -> [ENC_SEQ, DEC_HID]
    batch_H_proj = batch_H @ w["i2h_w"]
    # h2h: prev_hidden @ h2h_w + h2h_b -> [DEC_HID], 再 unsqueeze
    prev_hidden_proj = (prev_hidden @ w["h2h_w"] + w["h2h_b"])[None, :]
    res = np.tanh(batch_H_proj + prev_hidden_proj)
    e = res @ w["score_w"]          # [ENC_SEQ, 1]
    alpha = softmax(e, axis=0)      # [ENC_SEQ, 1]
    alpha_t = alpha.T               # [1, ENC_SEQ]
    context = (alpha_t @ batch_H).squeeze(0)  # [ENC_FEAT]
    concat = np.concatenate([context, char_onehot])  # [ENC_FEAT + NUM_CLS]

    # GRU: gates = concat @ gru_wx + gru_bx  +  prev_hidden @ gru_wh + gru_bh
    gates_x = concat @ w["gru_wx"] + w["gru_bx"]       # [3*DEC_HID]
    gates_h = prev_hidden @ w["gru_wh"] + w["gru_bh"]  # [3*DEC_HID]
    r = sigmoid(gates_x[0:DEC_HID] + gates_h[0:DEC_HID])
    z = sigmoid(gates_x[DEC_HID:2*DEC_HID] + gates_h[DEC_HID:2*DEC_HID])
    n = tanh(gates_x[2*DEC_HID:3*DEC_HID] + r * gates_h[2*DEC_HID:3*DEC_HID])
    h_next = (prev_hidden - n) * z + n
    return h_next


def reference_forward(w, enc_feat, max_steps=6, eos=49):
    """
    完整自回归解码, 生成 reference。
    enc_feat: [ENC_SEQ, ENC_FEAT]
    返回: structure_probs [steps, NUM_CLS], loc_preds [steps, LOC_NUM], tokens [steps]
    """
    hidden = np.zeros(DEC_HID, dtype=np.float32)
    pre_char = 0  # 起始 token (<start> = 0)
    tokens = []
    structs = []
    locs = []

    for i in range(max_steps):
        char_onehot = np.zeros(NUM_CLS, dtype=np.float32)
        char_onehot[pre_char] = 1.0
        hidden = attention_gru_cell(hidden, enc_feat, char_onehot, w)

        # structure_generator
        h = hidden @ w["fc1_w"] + w["fc1_b"]
        struct_logit = h @ w["struct_w"] + w["struct_b"]
        struct_prob = softmax(struct_logit)
        pre_char = int(np.argmax(struct_logit))

        # loc_generator
        h2 = hidden @ w["fc2_w"] + w["fc2_b"]
        loc = sigmoid(h2 @ w["loc_w"] + w["loc_b"])

        tokens.append(pre_char)
        structs.append(struct_prob)
        locs.append(loc)

        if pre_char == eos:
            break

    return np.array(structs), np.array(locs), np.array(tokens)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    w = make_weights()

    # 模拟 encoder 输出: [ENC_SEQ, ENC_FEAT] 随机特征
    rng = np.random.RandomState(7)
    enc_feat = (rng.randn(ENC_SEQ, ENC_FEAT) * 0.5).astype(np.float32)

    # ---- 写出权重 (顺序与 main.cpp 的 tsvec 索引一致) ----
    weights = [
        w["i2h_w"], w["h2h_w"], w["h2h_b"], w["score_w"],
        w["gru_wx"], w["gru_wh"], w["gru_bx"], w["gru_bh"],
        w["fc1_w"], w["fc1_b"], w["struct_w"], w["struct_b"],
        w["fc2_w"], w["fc2_b"], w["loc_w"], w["loc_b"],
    ]
    nplist_to_file(weights, os.path.join(here, "weights.antlist"))

    # ---- 写出 encoder 特征输入 ----
    np_to_file(enc_feat, os.path.join(here, "enc_feat.antts"))

    # ---- 计算参考输出 ----
    structs, locs, tokens = reference_forward(w, enc_feat, max_steps=6)
    np.save(os.path.join(here, "ref_structs.npy"), structs)
    np.save(os.path.join(here, "ref_locs.npy"), locs)
    np.save(os.path.join(here, "ref_tokens.npy"), tokens)

    print("生成完成:")
    print("  weights.antlist : %d 个权重张量" % len(weights))
    print("  enc_feat.antts  : encoder 特征 [%d, %d]" % (ENC_SEQ, ENC_FEAT))
    print("  参考 tokens     : %s" % str(tokens.tolist()))
    print("  参考 structs    : shape %s" % str(structs.shape))
    print("  参考 locs       : shape %s" % str(locs.shape))


if __name__ == "__main__":
    main()
