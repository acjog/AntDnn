#pragma once
#include "Tensor.h"
#include "Convolution.h"
namespace antdnn
{
	// 深度可分离卷积 / 分组卷积
	// 权重布局与 Conv2D 一致: [kh, kw, cin, cout]
	//   - 纯 depthwise: group == cin == cout, 每个输入通道用独立的单通道 kernel
	//   - 一般化分组卷积: group < cin, 每个 group 内做标准卷积
	// 输入/输出均为 NHWC: [H, W, C]
	DLL_ANTDNN_API void Conv2DDepthWise(
		Tensor &in_tensor, Tensor &out_tensor,
		Tensor &weights, Tensor &bias,
		int group,
		int padding_type = PADDING_VALID,
		int stride_h = 1, int stride_w = 1);
}
