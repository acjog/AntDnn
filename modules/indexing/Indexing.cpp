#include "Indexing.h"
#include <assert.h>

void antdnn::one_hot(Tensor &in_ts, Tensor &out_ts, int num_classes)
{
	// 输入 [N] (存储 float, 按 int 语义解释索引)
	int N = in_ts.size();
	int outshape[2] = { N, num_classes };
	Tensor out_tensor(2, outshape);
	out_tensor.set_to(0);

	const float *ptr_in = in_ts.ptr_read();
	float *ptr_out = out_tensor.ptr_write();

	for (int i = 0; i < N; i++)
	{
		int cls = (int)ptr_in[i];
		assert(cls >= 0 && cls < num_classes);
		ptr_out[i * num_classes + cls] = 1.0f;
	}
	out_ts = out_tensor;
}

void antdnn::argmax(Tensor &in_ts, Tensor &out_ts)
{
	int dim = in_ts.dim();
	const int *inshape = in_ts.shape();
	int D = inshape[dim - 1]; // 最后一维

	int outer = in_ts.size() / D;
	int outshape[8];
	for (int d = 0; d < dim - 1; d++)
		outshape[d] = inshape[d];
	int out_dim = dim - 1;
	if (out_dim == 0)
	{
		// 输入是 1 维, 输出标量 -> 用 1 维 size=1 表示
		out_dim = 1;
		outshape[0] = 1;
	}

	Tensor out_tensor(out_dim, outshape);
	const float *ptr_in = in_ts.ptr_read();
	float *ptr_out = out_tensor.ptr_write();

	for (int i = 0; i < outer; i++)
	{
		const float *row = ptr_in + i * D;
		int best = 0;
		float best_val = row[0];
		for (int j = 1; j < D; j++)
		{
			if (row[j] > best_val)
			{
				best_val = row[j];
				best = j;
			}
		}
		ptr_out[i] = (float)best;
	}
	out_ts = out_tensor;
}
