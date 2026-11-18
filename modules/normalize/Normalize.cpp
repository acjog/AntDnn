#include "Normalize.h"
#include <math.h>
#include <assert.h>

void antdnn::BatchNorm(
	Tensor &in_ts, Tensor &out_ts,
	Tensor &mean, Tensor &var,
	Tensor &scale, Tensor &bias,
	float eps)
{
	const int *inshape = in_ts.shape();
	int C = inshape[in_ts.dim() - 1];
	assert(mean.size() == C);
	assert(var.size() == C);

	bool has_scale = (scale.size() == C);
	bool has_bias = (bias.size() == C);

	Tensor out_tensor(in_ts.dim(), inshape);
	const float *ptr_in = in_ts.ptr_read();
	const float *ptr_mean = mean.ptr_read();
	const float *ptr_var = var.ptr_read();
	const float *ptr_scale = has_scale ? scale.ptr_read() : nullptr;
	const float *ptr_bias = has_bias ? bias.ptr_read() : nullptr;
	float *ptr_out = out_tensor.ptr_write();

	int total = in_ts.size();
	int spatial = total / C;

	for (int i = 0; i < spatial; i++)
	{
		for (int c = 0; c < C; c++)
		{
			float inv = 1.0f / sqrtf(ptr_var[c] + eps);
			float s = has_scale ? ptr_scale[c] : 1.0f;
			float b = has_bias ? ptr_bias[c] : 0.0f;
			float v = (*ptr_in - ptr_mean[c]) * inv * s + b;
			*ptr_out = v;
			ptr_in++;
			ptr_out++;
		}
	}
	out_ts = out_tensor;
}

void antdnn::GlobalAvgPool2D(Tensor & in_ts, Tensor & out_ts)
{
	const int *inshape = in_ts.shape();
	assert(in_ts.dim() == 3);
	int H = inshape[0];
	int W = inshape[1];
	int C = inshape[2];

	int outshape[3] = { 1, 1, C };
	Tensor out_tensor(3, outshape);
	out_tensor.set_to(0);

	const float *ptr_in = in_ts.ptr_read();
	float *ptr_out = out_tensor.ptr_write();

	int spatial = H * W;
	for (int i = 0; i < spatial; i++)
	{
		for (int c = 0; c < C; c++)
		{
			ptr_out[c] += ptr_in[c];
		}
		ptr_in += C;
	}
	float inv = 1.0f / (float)spatial;
	for (int c = 0; c < C; c++)
	{
		ptr_out[c] *= inv;
	}
	out_ts = out_tensor;
}

void antdnn::multiply(Tensor & in1_ts, Tensor & in2_ts, Tensor & out_ts)
{
	assert(in1_ts.size() == in2_ts.size());
	Tensor out_tensor(in1_ts.dim(), in1_ts.shape());
	const float *ptr1 = in1_ts.ptr_read();
	const float *ptr2 = in2_ts.ptr_read();
	float *ptr_out = out_tensor.ptr_write();

	int total = in1_ts.size();
	for (int i = 0; i < total; i++)
	{
		ptr_out[i] = ptr1[i] * ptr2[i];
	}
	out_ts = out_tensor;
}
