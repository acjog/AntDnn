#pragma once
#include <vector>
#include <antdnn/Tensor.h>

// PPLCNet backbone (SLANet_plus 的 Backbone)
// 输入 NHWC [H, W, 3], 输出 4 个尺度的特征 (NHWC)
// 权重从 .antlist 顺序读取, 与 export_encoder.py 的导出顺序一致
namespace slanet {

struct PPLCNet
{
    // 权重引用 (指向外部加载的 Tensor 数组)
    std::vector<antdnn::Tensor>* w;

    // 权重读取游标
    int cursor;

    PPLCNet(std::vector<antdnn::Tensor>* weights)
        : w(weights), cursor(0) {}

    // 取下一个权重
    antdnn::Tensor& next() { return (*w)[cursor++]; }

    // 前向: 输入 [H, W, 3] -> 输出 outs[4] (NHWC)
    // outs 尺寸: [122,122,64], [61,61,128], [31,31,256], [16,16,512]
    void forward(antdnn::Tensor& input, std::vector<antdnn::Tensor>& outs);
};

} // namespace slanet
