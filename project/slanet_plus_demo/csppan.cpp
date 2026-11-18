#include "csppan.h"
#include <antdnn/Convolution.h>
#include <antdnn/ConvolutionDepthWise.h>
#include <antdnn/Normalize.h>
#include <antdnn/Activation.h>
#include <antdnn/Shape.h>

namespace slanet {

// ConvBNLayer (act=hard_swish): conv -> bn -> hardswish
// 权重: conv_w, bn_mean, bn_var, bn_scale, bn_bias
static void conv_bn_hs(
    CSPPAN& net,
    antdnn::Tensor& in, antdnn::Tensor& out,
    int stride_h, int stride_w)
{
    antdnn::Tensor& conv_w = net.next();
    antdnn::Tensor& bn_mean = net.next();
    antdnn::Tensor& bn_var = net.next();
    antdnn::Tensor& bn_scale = net.next();
    antdnn::Tensor& bn_bias = net.next();

    antdnn::Tensor bias_dummy;
    antdnn::Conv2D(in, out, conv_w, bias_dummy, antdnn::PADDING_SAME, stride_h, stride_w);
    antdnn::BatchNorm(out, out, bn_mean, bn_var, bn_scale, bn_bias);
    antdnn::activation_hardswish(out);
}

// DPModule (act=hard_swish): dw(BN+HS) -> pw(BN+HS)
// 权重: dw_w, dw_bn_mean, dw_bn_var, dw_bn_scale, dw_bn_bias,
//       pw_w, pw_bn_mean, pw_bn_var, pw_bn_scale, pw_bn_bias
static void dp_module(
    CSPPAN& net,
    antdnn::Tensor& in, antdnn::Tensor& out,
    int kernel, int stride)
{
    antdnn::Tensor& dw_w = net.next();
    antdnn::Tensor& dw_bn_mean = net.next();
    antdnn::Tensor& dw_bn_var = net.next();
    antdnn::Tensor& dw_bn_scale = net.next();
    antdnn::Tensor& dw_bn_bias = net.next();

    int cin = in.shape()[2];
    antdnn::Tensor bias_dummy, tmp;
    antdnn::Conv2DDepthWise(in, tmp, dw_w, bias_dummy, cin, antdnn::PADDING_SAME, stride, stride);
    antdnn::BatchNorm(tmp, tmp, dw_bn_mean, dw_bn_var, dw_bn_scale, dw_bn_bias);
    antdnn::activation_hardswish(tmp);

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

// CSPLayer (use_depthwise=True, num_blocks=1, act=hard_swish, add_identity=False)
// 权重顺序:
//   short_conv(1x1 in->48) : conv_w + bn*4
//   main_conv(1x1 in->48)  : conv_w + bn*4
//   block.conv1(1x1 48->48): conv_w + bn*4
//   block.conv2(DPModule 5x5 48->48): dw_w + bn*4 + pw_w + bn*4
//   final_conv(1x1 96->96) : conv_w + bn*4
static void csp_layer(
    CSPPAN& net,
    antdnn::Tensor& in, antdnn::Tensor& out)
{
    int in_c = in.shape()[2];  // 192 (= 2*96)
    int mid = 48;              // out_channels(96) * expand_ratio(0.5) = 48

    // short_conv: 1x1 in_c -> mid
    antdnn::Tensor x_short;
    conv_bn_hs(net, in, x_short, 1, 1);

    // main_conv: 1x1 in_c -> mid
    antdnn::Tensor x_main;
    conv_bn_hs(net, in, x_main, 1, 1);

    // block: DarknetBottleneck(mid -> mid, kernel=5, expansion=1.0, use_depthwise=True, add_identity=False)
    //   conv1: 1x1 mid -> mid
    antdnn::Tensor b1;
    conv_bn_hs(net, x_main, b1, 1, 1);
    //   conv2: DPModule 5x5 mid -> mid (stride=1)
    antdnn::Tensor b2;
    dp_module(net, b1, b2, 5, 1);
    x_main = b2;

    // concat(x_main, x_short) along channel
    antdnn::Tensor* cat_inputs[2] = { &x_main, &x_short };
    antdnn::Tensor x_final_in;
    antdnn::concat(2, cat_inputs, 2, x_final_in);  // axis=2 (NHWC 通道维)

    // final_conv: 1x1 (2*mid) -> out_c
    antdnn::Tensor x_final;
    conv_bn_hs(net, x_final_in, x_final, 1, 1);

    out = x_final;
}

void CSPPAN::forward(std::vector<antdnn::Tensor>& inputs, std::vector<antdnn::Tensor>& outs)
{
    const int N = inputs.size();  // 4

    // ---- conv_t (Channel_T): 每个尺度 1x1 统一到 96 通道 ----
    std::vector<antdnn::Tensor> conv_t_outs(N);
    for (int i = 0; i < N; i++)
    {
        conv_bn_hs(*this, inputs[i], conv_t_outs[i], 1, 1);
    }

    // ---- top-down path ----
    std::vector<antdnn::Tensor> inner_outs;
    inner_outs.push_back(conv_t_outs[N - 1]);
    for (int idx = N - 1; idx >= 1; idx--)
    {
        // 上采样 inner_outs[0] 到 conv_t_outs[idx-1] 的尺寸
        antdnn::Tensor feat_high = inner_outs[0];
        antdnn::Tensor feat_low = conv_t_outs[idx - 1];
        int target_h = feat_low.shape()[0];
        int target_w = feat_low.shape()[1];
        antdnn::Tensor upsample_feat;
        antdnn::UpsampleNearest2D(feat_high, upsample_feat, 2.0f, 2.0f, target_h, target_w);

        // concat(upsample_feat, feat_low) along channel
        antdnn::Tensor* cat_inputs[2] = { &upsample_feat, &feat_low };
        antdnn::Tensor cat;
        antdnn::concat(2, cat_inputs, 2, cat);

        // CSPLayer
        antdnn::Tensor inner_out;
        csp_layer(*this, cat, inner_out);
        inner_outs.insert(inner_outs.begin(), inner_out);
    }

    // ---- bottom-up path ----
    std::vector<antdnn::Tensor> bottom_outs;
    bottom_outs.push_back(inner_outs[0]);
    for (int idx = 0; idx < N - 1; idx++)
    {
        antdnn::Tensor feat_low = bottom_outs.back();
        antdnn::Tensor feat_high = inner_outs[idx + 1];

        // downsample: DPModule 5x5 stride=2
        antdnn::Tensor downsample_feat;
        dp_module(*this, feat_low, downsample_feat, 5, 2);

        // concat(downsample_feat, feat_high) along channel
        antdnn::Tensor* cat_inputs[2] = { &downsample_feat, &feat_high };
        antdnn::Tensor cat;
        antdnn::concat(2, cat_inputs, 2, cat);

        // CSPLayer
        antdnn::Tensor out;
        csp_layer(*this, cat, out);
        bottom_outs.push_back(out);
    }

    outs = bottom_outs;
}

} // namespace slanet
