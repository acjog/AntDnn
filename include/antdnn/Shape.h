#pragma once
#include "Tensor.h"
namespace antdnn
{
	// 沿指定轴拼接, 输入必须除 axis 外各维相同
	// 输入为二维数组指针 + 数量; axis 为拼接维度 (0 起始)
	DLL_ANTDNN_API void concat(int ts_amount, Tensor **invec, int axis, Tensor &out_ts);

	// 最近邻上采样 (NHWC [H, W, C])
	// scale_h/scale_w 为目标尺度倍数; 若 out_h/out_w > 0 则按指定输出尺寸
	DLL_ANTDNN_API void UpsampleNearest2D(
		Tensor &in_ts, Tensor &out_ts,
		float scale_h, float scale_w,
		int out_h = 0, int out_w = 0);

	// 转置: 按 axes 重排维度 (axes 长度 = dim)
	DLL_ANTDNN_API void transpose(Tensor &in_ts, Tensor &out_ts, const int *axes);

	// 在指定位置插入长度为 1 的维度
	DLL_ANTDNN_API void unsqueeze(Tensor &in_ts, Tensor &out_ts, int axis);

	// 移除长度为 1 的维度 (axes 为要移除的维度列表, 长度 = axes_len)
	DLL_ANTDNN_API void squeeze(Tensor &in_ts, Tensor &out_ts, const int *axes, int axes_len);

	// 通用 reshape (元素总数不变), shape 为目标形状
	DLL_ANTDNN_API void reshape(Tensor &in_ts, Tensor &out_ts, int dim, const int *shape);
}
