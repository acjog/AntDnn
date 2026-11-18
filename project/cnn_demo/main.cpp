#include <antdnn/Tensor.h>
#include <antdnn/Convolution.h>
#include <antdnn/Activation.h>
#include <antdnn/Pooling.h>
#include <antdnn/Dense.h>
#include <antdnn/Merge.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <string>

using namespace antdnn;
using namespace std;

// ---------------------------------------------------------------------------
// 一个完整可运行的 CNN 前向推理 demo
//
// 网络结构 (与 gen_weights.py 中一致):
//   输入   : 1x8x8x1 (NHWC, 单通道灰度图)
//   Conv2D : 3x3, 1 -> 4, PADDING_SAME  + ReLU
//   Pool   : MaxPooling2D 2x2          -> 4x4x4
//   Conv2D : 3x3, 4 -> 8, PADDING_VALID + ReLU
//   Pool   : MaxPooling2D 2x2          -> 1x1x8
//   Flatten
//   Dense  : 8 -> 3
//   Softmax
// ---------------------------------------------------------------------------

// 从文件读取参考输出(第一行, 空格分隔的 float)
static bool load_ref_output(const string &path, vector<float> &out)
{
    ifstream f(path);
    if (!f.is_open())
        return false;
    float v;
    while (f >> v)
        out.push_back(v);
    return !out.empty();
}

int main()
{
    // ---- 1. 加载权重 ----
    vector<Tensor> tsvec;
    load_tensor_vector("weights.antlist", tsvec);
    if (tsvec.size() != 6)
    {
        cerr << "权重加载失败或数量不对 (expect 6, got " << tsvec.size() << ")." << endl;
        cerr << "请先运行: python3 gen_weights.py" << endl;
        return 1;
    }

    // ---- 2. 加载输入图像 ----
    Tensor in_ts;
    if (!in_ts.load_file("img.antts"))
    {
        cerr << "输入图像加载失败, 请先运行: python3 gen_weights.py" << endl;
        return 1;
    }
    cout << "输入 shape: " << in_ts.shape()[0] << "x"
         << in_ts.shape()[1] << "x"
         << in_ts.shape()[2] << " (HxWxC)" << endl;

    // ---- 3. 前向推理 ----
    Tensor out_ts;

    // Conv2D(3x3, 1->4, SAME) + ReLU
    Conv2D(in_ts, out_ts, tsvec[0], tsvec[1], PADDING_SAME);
    activation_relu(out_ts);
    // MaxPooling2D 2x2
    MaxPooling2D(out_ts, out_ts, 2, 2);

    // Conv2D(3x3, 4->8, VALID) + ReLU
    Conv2D(out_ts, out_ts, tsvec[2], tsvec[3], PADDING_VALID);
    activation_relu(out_ts);
    // MaxPooling2D 2x2
    MaxPooling2D(out_ts, out_ts, 2, 2);

    // Flatten
    flatten(out_ts);

    // Dense(8 -> 3) + Softmax
    Dense(out_ts, out_ts, tsvec[4], tsvec[5]);
    activation_softmax(out_ts);

    // ---- 4. 输出结果 ----
    cout << "\n最终输出 (softmax 概率, 3 类):" << endl;
    const float *ptr = out_ts.ptr_read();
    int n = out_ts.size();
    for (int i = 0; i < n; i++)
    {
        cout << "  class " << i << ": " << ptr[i] << endl;
    }

    // 找最大类
    int argmax = 0;
    for (int i = 1; i < n; i++)
        if (ptr[i] > ptr[argmax])
            argmax = i;
    cout << "预测类别: " << argmax << endl;

    // ---- 5. 与 numpy 参考输出比对 ----
    vector<float> ref;
    if (load_ref_output("ref_output.txt", ref))
    {
        cout << "\n===== 与 numpy 参考输出比对 =====" << endl;
        if (ref.size() != (size_t)n)
        {
            cout << "参考输出数量不匹配 (" << ref.size() << " vs " << n << ")" << endl;
        }
        else
        {
            double max_diff = 0;
            for (int i = 0; i < n; i++)
                max_diff = max(max_diff, (double)fabs(ptr[i] - ref[i]));
            cout << "最大误差: " << max_diff << endl;
            if (max_diff < 1e-3)
                cout << "=== 结果一致, DEMO 通过 === " << endl;
            else
                cout << "误差偏大, 请检查实现" << endl;
        }
    }
    else
    {
        cout << "\n(未找到 ref_output.txt, 跳过数值比对; "
                "运行 gen_weights.py 会生成该文件)" << endl;
    }

    return 0;
}
