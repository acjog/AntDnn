#include "pplcnet.h"
#include <antdnn/Convolution.h>
#include <antdnn/ConvolutionDepthWise.h>
#include <antdnn/Normalize.h>
#include <antdnn/Activation.h>

namespace slanet {

// ConvBNLayer: conv -> bn -> hardswish
// 权重: conv_w, bn_mean, bn_var, bn_scale, bn_bias (共5个, 按此顺序读取)
static void conv_bn_hardswish(
    antdnn::Tensor& in, antdnn::Tensor& out,
    antdnn::Tensor& conv_w, antdnn::Tensor& bn_mean,
    antdnn::Tensor& bn_var, antdnn::Tensor& bn_scale,
    antdnn::Tensor& bn_bias,
    int stride_h, int stride_w, antdnn::Tensor& bias_dummy)
{
    // conv (无 bias, 但 AntDnn Conv2D 需要 bias 参数, 传一个全0)
    antdnn::Conv2D(in, out, conv_w, bias_dummy, antdnn::PADDING_SAME, stride_h, stride_w);
    // bn
    antdnn::BatchNorm(out, out, bn_mean, bn_var, bn_scale, bn_bias);
    // hardswish
    antdnn::activation_hardswish(out);
}

// DepthwiseSeparable: dw_conv(BN+HS) -> [SE] -> pw_conv(BN+HS)
// 权重: dw_w, dw_bn_mean, dw_bn_var, dw_bn_scale, dw_bn_bias,
//       [se_fc1_w, se_fc2_w (若 use_se)],
//       pw_w, pw_bn_mean, pw_bn_var, pw_bn_scale, pw_bn_bias
static void depthwise_separable(
    PPLCNet& net,
    antdnn::Tensor& in, antdnn::Tensor& out,
    int dw_size, int stride, bool use_se, antdnn::Tensor& bias_dummy)
{
    // depthwise conv (groups = channels)
    antdnn::Tensor& dw_w = net.next();
    antdnn::Tensor& dw_bn_mean = net.next();
    antdnn::Tensor& dw_bn_var = net.next();
    antdnn::Tensor& dw_bn_scale = net.next();
    antdnn::Tensor& dw_bn_bias = net.next();

    int cin = in.shape()[2];
    antdnn::Tensor tmp;
    antdnn::Conv2DDepthWise(in, tmp, dw_w, bias_dummy, cin, antdnn::PADDING_SAME, stride, stride);
    antdnn::BatchNorm(tmp, tmp, dw_bn_mean, dw_bn_var, dw_bn_scale, dw_bn_bias);
    antdnn::activation_hardswish(tmp);

    // SE (可选)
    if (use_se)
    {
        antdnn::Tensor& se_fc1_w = net.next(); // [1,1,cin,cin/4]
        antdnn::Tensor& se_fc1_b = net.next(); // [cin/4]
        antdnn::Tensor& se_fc2_w = net.next(); // [1,1,cin/4,cin]
        antdnn::Tensor& se_fc2_b = net.next(); // [cin]
        int C = tmp.shape()[2];
        antdnn::Tensor gap, fc1, fc2, se_out;
        antdnn::GlobalAvgPool2D(tmp, gap);   // [1,1,C]
        // fc1: 1x1 C->red, +bias, relu
        antdnn::Conv2D(gap, fc1, se_fc1_w, se_fc1_b, antdnn::PADDING_VALID);
        antdnn::activation_relu(fc1);
        // fc2: 1x1 red->C, +bias, hardsigmoid
        antdnn::Conv2D(fc1, fc2, se_fc2_w, se_fc2_b, antdnn::PADDING_VALID);
        antdnn::activation_hardsigmoid(fc2);
        // channel-wise 广播乘法: tmp[H,W,C] * fc2[1,1,C]
        {
            int H = tmp.shape()[0], W = tmp.shape()[1];
            int Cch = tmp.shape()[2];
            const float* scale = fc2.ptr_read();   // [C]
            se_out.recreate(3, tmp.shape());
            const float* src = tmp.ptr_read();
            float* dst = se_out.ptr_write();
            for (int i = 0; i < H * W; i++)
                for (int c = 0; c < Cch; c++)
                    dst[i * Cch + c] = src[i * Cch + c] * scale[c];
        }
        tmp = se_out;
    }

    // pointwise conv (1x1)
    antdnn::Tensor& pw_w = net.next();
    antdnn::Tensor& pw_bn_mean = net.next();
    antdnn::Tensor& pw_bn_var = net.next();
    antdnn::Tensor& pw_bn_scale = net.next();
    antdnn::Tensor& pw_bn_bias = net.next();

    antdnn::Tensor pw_out;
    antdnn::Conv2D(tmp, pw_out, pw_w, bias_dummy, antdnn::PADDING_VALID);
    antdnn::BatchNorm(pw_out, pw_out, pw_bn_mean, pw_bn_var, pw_bn_scale, pw_bn_bias);
    antdnn::activation_hardswish(pw_out);

    out = pw_out;
}

void PPLCNet::forward(antdnn::Tensor& input, std::vector<antdnn::Tensor>& outs)
{
    // bias_dummy: 全 0 (PPLCNet 的 conv 都是 bias_attr=False)
    // 用一个空 Tensor 传给 Conv2D 作为 bias
    antdnn::Tensor bias_dummy;

    // ---- conv1: 3x3, 3->16, stride=2 ----
    antdnn::Tensor x;
    {
        antdnn::Tensor& conv1_w = next();
        antdnn::Tensor& bn_mean = next();
        antdnn::Tensor& bn_var = next();
        antdnn::Tensor& bn_scale = next();
        antdnn::Tensor& bn_bias = next();
        conv_bn_hardswish(input, x, conv1_w, bn_mean, bn_var, bn_scale, bn_bias, 2, 2, bias_dummy);
    }

    // ---- blocks2: [k=3, 16->32, s=1, no_se] ----
    antdnn::Tensor x2;
    depthwise_separable(*this, x, x2, 3, 1, false, bias_dummy);
    x = x2;

    // ---- blocks3: [3,32->64,s=2], [3,64->64,s=1] ----
    antdnn::Tensor x3;
    depthwise_separable(*this, x, x3, 3, 2, false, bias_dummy);
    x = x3;
    depthwise_separable(*this, x, x3, 3, 1, false, bias_dummy);
    x = x3;
    outs.push_back(x);  // 尺度1: [122,122,64]

    // ---- blocks4: [3,64->128,s=2], [3,128->128,s=1] ----
    antdnn::Tensor x4;
    depthwise_separable(*this, x, x4, 3, 2, false, bias_dummy);
    x = x4;
    depthwise_separable(*this, x, x4, 3, 1, false, bias_dummy);
    x = x4;
    outs.push_back(x);  // 尺度2: [61,61,128]

    // ---- blocks5: [3,128->256,s=2], 5x[5,256->256,s=1] ----
    antdnn::Tensor x5;
    depthwise_separable(*this, x, x5, 3, 2, false, bias_dummy);
    x = x5;
    for (int i = 0; i < 5; i++)
    {
        depthwise_separable(*this, x, x5, 5, 1, false, bias_dummy);
        x = x5;
    }
    outs.push_back(x);  // 尺度3: [31,31,256]

    // ---- blocks6: [5,256->512,s=2,se], [5,512->512,s=1,se] ----
    antdnn::Tensor x6;
    depthwise_separable(*this, x, x6, 5, 2, true, bias_dummy);
    x = x6;
    depthwise_separable(*this, x, x6, 5, 1, true, bias_dummy);
    x = x6;
    outs.push_back(x);  // 尺度4: [16,16,512]
}

} // namespace slanet
