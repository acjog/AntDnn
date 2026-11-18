# -*- coding: utf-8 -*-
"""
把 SLANet_plus 的真实 decoder 权重 (.bin) 转换成 AntDnn 的 .antlist 格式。

这些 .bin 由 export_slanet_plus_encoder.py 从 ONNX 的 Loop body 中提取,
每个文件是 fp32 的裸数据 (无 header), 已按 AntDnn 所需的布局存储:

  dec_linear0_w  [96, 256]   = i2h   (key proj)
  dec_linear1_w  [256, 256]  = h2h
  dec_linear1_b  [256,]      = h2h bias
  dec_linear2_w  [256, 1]    = score
  dec_gru_wx     [146, 768]  = GRU W_x (已转置)
  dec_gru_wh     [256, 768]  = GRU W_h (已转置)
  dec_gru_bx     [768,]      = GRU b_x
  dec_gru_bh     [768,]      = GRU b_h
  dec_linear3_w  [256, 256]  = structure_generator fc1
  dec_linear3_b  [256,]
  dec_linear4_w  [256, 50]   = structure_generator fc2 (struct logits)
  dec_linear4_b  [50,]
  dec_linear5_w  [256, 256]  = loc_generator fc1
  dec_linear5_b  [256,]
  dec_linear6_w  [256, 8]    = loc_generator fc2 (loc)
  dec_linear6_b  [8,]

用法:
  python3 export_decoder.py [SLANet_plus_dir]

输出: weights.antlist (当前目录)
"""

import numpy as np
import os
import sys

# 16 个权重: (文件名, 期望 shape)
WEIGHTS = [
    ("dec_linear0_w", (96, 256)),
    ("dec_linear1_w", (256, 256)),
    ("dec_linear1_b", (256,)),
    ("dec_linear2_w", (256, 1)),
    ("dec_gru_wx",    (146, 768)),
    ("dec_gru_wh",    (256, 768)),
    ("dec_gru_bx",    (768,)),
    ("dec_gru_bh",    (768,)),
    ("dec_linear3_w", (256, 256)),
    ("dec_linear3_b", (256,)),
    ("dec_linear4_w", (256, 50)),
    ("dec_linear4_b", (50,)),
    ("dec_linear5_w", (256, 256)),
    ("dec_linear5_b", (256,)),
    ("dec_linear6_w", (256, 8)),
    ("dec_linear6_b", (8,)),
]


def nplist_to_file(nplist, filename):
    """内联实现 antdnn 的 .antlist 写出 (避免路径耦合)"""
    with open(filename, "wb") as f:
        f.write(b"ant\0")          # file flag
        f.write(b"lis\0")          # data type
        f.write(np.array([len(nplist)], dtype=np.int32).tobytes())
        for npdata in nplist:
            f.write(b"tsd\0")
            header = [len(npdata.shape)] + list(npdata.shape)
            f.write(np.array(header, dtype=np.int32).tobytes())
            f.write(npdata.astype(np.float32).tobytes())


def main():
    # 默认 SLANet_plus 目录
    if len(sys.argv) > 1:
        src_dir = sys.argv[1]
    else:
        src_dir = os.environ.get(
            "SLANET_PLUS_DIR",
            "/Users/acjog/Documents/code/ncnn-webassembly-ppocr/SLANet_plus",
        )

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(out_dir, "weights.antlist")

    arrays = []
    for name, shape in WEIGHTS:
        path = os.path.join(src_dir, name + ".bin")
        if not os.path.exists(path):
            print(f"错误: 找不到 {path}")
            sys.exit(1)
        # 裸 fp32 数据
        arr = np.fromfile(path, dtype=np.float32)
        expect = int(np.prod(shape))
        if arr.size != expect:
            print(f"错误: {name} 元素数 {arr.size} != 期望 {expect} (shape {shape})")
            sys.exit(1)
        arr = arr.reshape(shape)
        arrays.append(arr)
        print(f"  {name:16s} {str(shape):20s} OK")

    nplist_to_file(arrays, out_file)
    print(f"\n已生成 {out_file} ({len(arrays)} 个权重张量)")


if __name__ == "__main__":
    main()
