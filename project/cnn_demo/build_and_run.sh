#!/bin/bash
# 一键编译并运行 cnn_demo
# 用法:  ./build_and_run.sh
set -e

cd "$(dirname "$0")"

# 项目根目录 (cnn_demo 的上级目录)
ROOT="$(cd ../.. && pwd)"

# 1. 生成测试权重 (需要 python3 + numpy)
echo "==> 生成测试权重..."
python3 gen_weights.py

# 2. 编译
echo "==> 编译..."
clang++ -std=c++11 -O2 \
    -I"$ROOT/include" \
    -I"$ROOT/include/antdnn" \
    -I"$ROOT/3rdparty" \
     $(find "$ROOT/modules" -name '*.cpp') \
    main.cpp \
    -o cnn_demo

# 3. 运行
echo "==> 运行..."
./cnn_demo
