#!/bin/bash
# 一键编译并运行 slanet_plus_demo
# 用法:  ./build_and_run.sh
set -e

cd "$(dirname "$0")"

# 项目根目录 (slanet_plus_demo 的上级的上级)
ROOT="$(cd ../.. && pwd)"

# 1. 从 SLANet_plus_pretrained.pdparams 直接转换 encoder + decoder 权重
#    (需要 python3.12 + paddle; 不再依赖 onnx)
echo "==> 从 .pdparams 转换权重..."
if python3.12 export_pdparams.py 2>/dev/null; then
    echo "  encoder.antlist + weights.antlist 已生成"
else
    echo "  (.pdparams 转换失败, 回退到模拟权重)"
    python3 gen_weights.py
fi

# 2. 编译 (递归收集 modules 下所有 .cpp, 链接 opencv-mobile framework)
#    opencv-mobile 4.14.0 framework 是 fat binary (arm64 + x86_64), 可原生编译
echo "==> 编译..."
OCV_FW="$ROOT/3rdparty/opencv-mobile"
clang++ -std=c++11 -O2 \
    -I"$ROOT/include" \
    -I"$ROOT/include/antdnn" \
    -I"$ROOT/3rdparty" \
    -F"$OCV_FW" -framework opencv2 \
    $(find "$ROOT/modules" -name '*.cpp') \
    main.cpp pplcnet.cpp csppan.cpp \
    -o slanet_plus_demo

# 3. 运行
echo "==> 运行..."
IMG="${1:-}"
if [ -n "$IMG" ]; then
    ./slanet_plus_demo "$IMG"
else
    ./slanet_plus_demo
fi
