#pragma once
#include <vector>
#include <antdnn/Tensor.h>

// CSPPAN neck (SLANet_plus 的 Neck)
// 输入 4 个尺度特征 (NHWC), 输出 4 个尺度特征 (NHWC, 通道统一为 96)
// 权重从 .antlist 顺序读取, 与 export_encoder.py 的导出顺序一致
namespace slanet {

struct CSPPAN
{
    std::vector<antdnn::Tensor>* w;
    int cursor;

    CSPPAN(std::vector<antdnn::Tensor>* weights, int start_cursor = 0)
        : w(weights), cursor(start_cursor) {}

    antdnn::Tensor& next() { return (*w)[cursor++]; }

    // 前向: 输入 4 尺度 (NHWC), 输出 4 尺度 (NHWC, 通道 96)
    // 输出最后一个尺度 [16,16,96] 会被展平成 [256, 96] 供 decoder 使用
    void forward(std::vector<antdnn::Tensor>& inputs, std::vector<antdnn::Tensor>& outs);
};

} // namespace slanet
