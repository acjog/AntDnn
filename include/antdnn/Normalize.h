#pragma once
#include "Tensor.h"
namespace antdnn
{
	// BatchNorm (推理期): y = (x - mean) / sqrt(var + eps) * scale + bias
	// 输入 NHWC [H, W, C], mean/var/scale/bias 均为 [C]
	// 若 scale/bias 为空 Tensor (size()==0), 则默认 scale=1, bias=0
	DLL_ANTDNN_API void BatchNorm(
		Tensor &in_ts, Tensor &out_ts,
		Tensor &mean, Tensor &var,
		Tensor &scale, Tensor &bias,
		float eps = 1e-5f);

	// 全局平均池化: 输入 [H, W, C] -> 输出 [1, 1, C]
	DLL_ANTDNN_API void GlobalAvgPool2D(Tensor & in_ts, Tensor & out_ts);

	// 逐元素乘: 两个同 size 张量对应元素相乘
	DLL_ANTDNN_API void multiply(Tensor & in1_ts, Tensor & in2_ts, Tensor & out_ts);
}
