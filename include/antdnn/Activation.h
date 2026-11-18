#pragma once
#include "Tensor.h"
namespace antdnn
{
	DLL_ANTDNN_API void activation_relu(Tensor & ts);
	DLL_ANTDNN_API void activation_softmax(Tensor & ts);
	DLL_ANTDNN_API void flatten(Tensor & ts);

	// 新增激活算子
	DLL_ANTDNN_API void activation_leaky_relu(Tensor & ts, float alpha = 0.1f);
	DLL_ANTDNN_API void activation_hardswish(Tensor & ts);
	DLL_ANTDNN_API void activation_hardsigmoid(Tensor & ts);
	DLL_ANTDNN_API void activation_tanh(Tensor & ts);
	DLL_ANTDNN_API void activation_sigmoid(Tensor & ts);
}