#include "Shape.h"
#include <assert.h>
#include <string.h>

// 计算从 dim 到末尾的累计乘积 (stride), 用于多维索引换算
static int calcSize(const int *shape, int dim)
{
	int s = 1;
	for (int i = 0; i < dim; i++)
		s *= shape[i];
	return s;
}

void antdnn::concat(int ts_amount, Tensor **invec, int axis, Tensor &out_ts)
{
	assert(ts_amount > 0);
	int dim = invec[0]->dim();
	assert(axis >= 0 && axis < dim);

	const int *shape0 = invec[0]->shape();
	int outshape[8]; // 支持最多 8 维
	for (int d = 0; d < dim; d++)
		outshape[d] = shape0[d];
	outshape[axis] = 0;
	for (int i = 0; i < ts_amount; i++)
	{
		assert(invec[i]->dim() == dim);
		for (int d = 0; d < dim; d++)
		{
			if (d == axis)
				outshape[d] += invec[i]->shape()[d];
			else
				assert(invec[i]->shape()[d] == shape0[d]);
		}
	}

	Tensor out_tensor(dim, outshape);
	float *ptr_out = out_tensor.ptr_write();

	// 计算 axis 前后各维的 stride
	int outer = 1;      // axis 之前各维乘积
	for (int d = 0; d < axis; d++)
		outer *= shape0[d];
	int inner = 1;      // axis 之后各维乘积
	for (int d = axis + 1; d < dim; d++)
		inner *= shape0[d];

	for (int o = 0; o < outer; o++)
	{
		for (int i = 0; i < ts_amount; i++)
		{
			int in_axis = invec[i]->shape()[axis];
			int copy_len = in_axis * inner;
			const float *ptr_in = invec[i]->ptr_read() + o * copy_len;
			memcpy(ptr_out, ptr_in, copy_len * sizeof(float));
			ptr_out += copy_len;
		}
	}
	out_ts = out_tensor;
}

void antdnn::UpsampleNearest2D(
	Tensor &in_ts, Tensor &out_ts,
	float scale_h, float scale_w,
	int out_h, int out_w)
{
	assert(in_ts.dim() == 3);
	const int *inshape = in_ts.shape();
	int H = inshape[0];
	int W = inshape[1];
	int C = inshape[2];

	int oH = out_h > 0 ? out_h : (int)(H * scale_h);
	int oW = out_w > 0 ? out_w : (int)(W * scale_w);

	int outshape[3] = { oH, oW, C };
	Tensor out_tensor(3, outshape);
	const float *ptr_in = in_ts.ptr_read();
	float *ptr_out = out_tensor.ptr_write();

	for (int i = 0; i < oH; i++)
	{
		int src_i = (int)(i / scale_h);
		if (src_i >= H) src_i = H - 1;
		for (int j = 0; j < oW; j++)
		{
			int src_j = (int)(j / scale_w);
			if (src_j >= W) src_j = W - 1;
			const float *src_pixel = ptr_in + (src_i * W + src_j) * C;
			memcpy(ptr_out, src_pixel, C * sizeof(float));
			ptr_out += C;
		}
	}
	out_ts = out_tensor;
}

void antdnn::transpose(Tensor &in_ts, Tensor &out_ts, const int *axes)
{
	int dim = in_ts.dim();
	const int *inshape = in_ts.shape();

	int outshape[8];
	for (int d = 0; d < dim; d++)
		outshape[d] = inshape[axes[d]];

	// 输入各维 stride (row-major)
	int in_stride[8];
	in_stride[dim - 1] = 1;
	for (int d = dim - 2; d >= 0; d--)
		in_stride[d] = in_stride[d + 1] * inshape[d + 1];

	Tensor out_tensor(dim, outshape);
	const float *ptr_in = in_ts.ptr_read();
	float *ptr_out = out_tensor.ptr_write();

	int total = in_ts.size();
	// 遍历输出线性索引, 反解出多维坐标
	int coords[8];
	for (int idx = 0; idx < total; idx++)
	{
		int rem = idx;
		for (int d = 0; d < dim; d++)
		{
			int s = 1;
			for (int e = d + 1; e < dim; e++)
				s *= outshape[e];
			coords[d] = rem / s;
			rem = rem % s;
		}
		// 根据 axes 反推输入坐标
		int in_idx = 0;
		for (int d = 0; d < dim; d++)
		{
			in_idx += coords[d] * in_stride[axes[d]];
		}
		ptr_out[idx] = ptr_in[in_idx];
	}
	out_ts = out_tensor;
}

void antdnn::unsqueeze(Tensor &in_ts, Tensor &out_ts, int axis)
{
	int dim = in_ts.dim();
	assert(axis >= 0 && axis <= dim);
	const int *inshape = in_ts.shape();

	int outshape[8];
	int k = 0;
	for (int d = 0; d <= dim; d++)
	{
		if (d == axis)
			outshape[d] = 1;
		else
			outshape[d] = inshape[k++];
	}

	Tensor out_tensor(dim + 1, outshape);
	const float *ptr_in = in_ts.ptr_read();
	float *ptr_out = out_tensor.ptr_write();
	memcpy(ptr_out, ptr_in, in_ts.size() * sizeof(float));
	out_ts = out_tensor;
}

void antdnn::squeeze(Tensor &in_ts, Tensor &out_ts, const int *axes, int axes_len)
{
	int dim = in_ts.dim();
	const int *inshape = in_ts.shape();

	// 标记要删除的维度
	bool del[8] = { false };
	for (int i = 0; i < axes_len; i++)
	{
		assert(axes[i] >= 0 && axes[i] < dim);
		assert(inshape[axes[i]] == 1);
		del[axes[i]] = true;
	}

	int outshape[8];
	int new_dim = 0;
	for (int d = 0; d < dim; d++)
	{
		if (!del[d])
			outshape[new_dim++] = inshape[d];
	}

	Tensor out_tensor(new_dim, outshape);
	const float *ptr_in = in_ts.ptr_read();
	float *ptr_out = out_tensor.ptr_write();
	memcpy(ptr_out, ptr_in, in_ts.size() * sizeof(float));
	out_ts = out_tensor;
}

void antdnn::reshape(Tensor &in_ts, Tensor &out_ts, int dim, const int *shape)
{
	assert(calcSize(shape, dim) == in_ts.size());
	Tensor out_tensor(dim, shape);
	const float *ptr_in = in_ts.ptr_read();
	float *ptr_out = out_tensor.ptr_write();
	memcpy(ptr_out, ptr_in, in_ts.size() * sizeof(float));
	out_ts = out_tensor;
}
