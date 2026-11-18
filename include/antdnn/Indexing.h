#pragma once
#include "Tensor.h"
namespace antdnn
{
	// OneHot: 输入 int 索引 [N], 输出 [N, num_classes] 的 one-hot 矩阵
	// (输入 Tensor 存储为 float, 但按 int 语义解释)
	DLL_ANTDNN_API void one_hot(Tensor &in_ts, Tensor &out_ts, int num_classes);

	// ArgMax: 沿最后一维取最大值的索引, 输入 [..., D] -> 输出 [...]
	DLL_ANTDNN_API void argmax(Tensor &in_ts, Tensor &out_ts);
}
