#include "ConvolutionDepthWise.h"
#include <assert.h>
#ifdef USE_optimization_OPENMP
#include <omp.h>
#endif
extern int cpu_num;

// 单次卷积累加: res[c] += mul_1 * mul2[c]
static inline void dw_mul_m_add(const float mul_1, const float *mul2, float *res, int count)
{
	for (int c = 0; c < count; c++)
		res[c] += mul_1 * mul2[c];
}

// 纯 depthwise 卷积单元
// 输入 NHWC [H, W, C], 权重 [kh, kw, C, 1] (每个通道独立 kernel)
// group == cin == cout
static void Conv2DDepthWise_unit(
	const float *srcptr, const float *kerptr, float *dstptr, const float *ptr_bias,
	const int calc_rows,
	const int *srcshape,   // [H, W, C]
	const int *dst_shape,  // [oH, oW, C]
	const int *ker_shape,  // [kh, kw, 1, C]
	int stride_h, int stride_w)
{
	const int H = srcshape[0];
	const int W = srcshape[1];
	const int C = srcshape[2];   // 输入输出通道数 (== cout)
	const int oH = dst_shape[0];
	const int oW = dst_shape[1];
	const int kh = ker_shape[0];
	const int kw = ker_shape[1];

	// 逐输出行
	for (int i = 0; i < calc_rows; i++)
	{
		for (int j = 0; j < oW; j++)
		{
			float *out_pixel = dstptr + (i * oW + j) * C;
			// 逐输出通道 (= 逐输入通道, depthwise)
			for (int c = 0; c < C; c++)
			{
				float acc = 0.0f;
				for (int ki = 0; ki < kh; ki++)
				{
					int ih = i * stride_h + ki;
					if (ih < 0 || ih >= H) continue;
					const float *src_row = srcptr + (ih * W) * C;
					for (int kj = 0; kj < kw; kj++)
					{
						int iw = j * stride_w + kj;
						if (iw < 0 || iw >= W) continue;
						// 权重 [kh, kw, 1, C]: 位置 (ki,kj,0,c) -> (ki*kw+kj)*C + c
						acc += kerptr[(ki * kw + kj) * C + c] * src_row[iw * C + c];
					}
				}
				out_pixel[c] = acc + ptr_bias[c];
			}
		}
	}
}

// 一般化分组卷积单元
// 每个 group 内部做标准卷积: inch_g -> outch_g
static void Conv2DGroup_unit(
	const float *srcptr, const float *kerptr, float *dstptr, const float *ptr_bias,
	const int calc_rows,
	const int *srcshape,   // [H, W, C]
	const int *dst_shape,  // [oH, oW, cout]
	const int *ker_shape,  // [kh, kw, cin, cout]
	const int group,
	int stride_h, int stride_w)
{
	const int H = srcshape[0];
	const int W = srcshape[1];
	const int C = srcshape[2];          // cin
	const int oH = dst_shape[0];
	const int oW = dst_shape[1];
	const int cout = dst_shape[2];
	const int kh = ker_shape[0];
	const int kw = ker_shape[1];

	const int inch_g = C / group;
	const int outch_g = cout / group;

	for (int i = 0; i < calc_rows; i++)
	{
		for (int j = 0; j < oW; j++)
		{
			float *out_pixel = dstptr + (i * oW + j) * cout;
			// 逐输出通道
			for (int co = 0; co < cout; co++)
			{
				int g = co / outch_g;
				float acc = 0.0f;
				// 该 group 的输入通道范围
				int ci_begin = g * inch_g;
				int ci_end = ci_begin + inch_g;
				for (int ci = ci_begin; ci < ci_end; ci++)
				{
					// 权重偏移: [kh, kw, ci, co]
					const float *k = kerptr + ((ci * cout) + co) * kh * kw;
					for (int ki = 0; ki < kh; ki++)
					{
						int ih = i * stride_h + ki;
						if (ih < 0 || ih >= H) continue;
						const float *src_row = srcptr + (ih * W) * C;
						for (int kj = 0; kj < kw; kj++)
						{
							int iw = j * stride_w + kj;
							if (iw < 0 || iw >= W) continue;
							acc += k[ki * kw + kj] * src_row[iw * C + ci];
						}
					}
				}
				out_pixel[co] = acc + ptr_bias[co];
			}
		}
	}
}

void antdnn::Conv2DDepthWise(
	Tensor &in_tensor, Tensor &out_ts,
	Tensor &weights, Tensor &bias,
	int group,
	int padding_type,
	int stride_h, int stride_w)
{
	auto wei_shape = weights.shape();  // [kh, kw, cin, cout]
	const int kh = wei_shape[0];
	const int kw = wei_shape[1];
	const int cin = wei_shape[2];
	const int cout = wei_shape[3];

	// depthwise: cin==1, cout==group; 分组卷积: cin%group==0, cout%group==0
	if (cin == 1)
		assert(cout == group);
	else
	{
		assert(cin % group == 0);
		assert(cout % group == 0);
	}

	// 支持无 bias 卷积: bias 为空时创建全 0 bias
	Tensor bias_zero;
	if (bias.size() == 0)
	{
		int bc[1] = { cout };
		bias_zero.recreate(1, bc);
		bias_zero.set_to(0);
	}
	Tensor &bias_ref = (bias.size() == 0) ? bias_zero : bias;

	Tensor mid_tensor = in_tensor;
	if (padding_type == PADDING_SAME)
	{
		int top = (kh - 1) / 2, bottom = top;
		int left = (kw - 1) / 2, right = left;
		mid_tensor.makeBorder2D(top, bottom, left, right, 0);
	}

	auto src_shape = mid_tensor.shape();
	int dst_shape[3] = {
		(src_shape[0] - kh) / stride_h + 1,
		(src_shape[1] - kw) / stride_w + 1,
		cout
	};
	Tensor out_tensor;
	out_tensor.recreate(3, dst_shape);
	out_tensor.set_to(0);

	const float *srcptr = mid_tensor.ptr_read();
	const float *kerptr = weights.ptr_read();
	const float *biasptr = bias_ref.ptr_read();
	float *dstptr = out_tensor.ptr_write();

	const bool is_depthwise = (cin == 1 && group == cout);

#ifdef USE_optimization_OPENMP
	int each_cpu_rows = dst_shape[0] / cpu_num;
	int last_rows = each_cpu_rows + dst_shape[0] % cpu_num;
	int each_cpu_rows_src = each_cpu_rows * stride_h * src_shape[1] * src_shape[2];
	int each_cpu_rows_dst = each_cpu_rows * dst_shape[1] * dst_shape[2];
#pragma omp parallel for
	for (int i = 0; i < cpu_num; i++)
	{
		float *dstptr_omp = dstptr + each_cpu_rows_dst * i;
		const float *srcptr_omp = srcptr + each_cpu_rows_src * i;
		int rows = (i < cpu_num - 1) ? each_cpu_rows : last_rows;
		if (is_depthwise)
			Conv2DDepthWise_unit(srcptr_omp, kerptr, dstptr_omp, biasptr, rows,
				src_shape, dst_shape, wei_shape, stride_h, stride_w);
		else
			Conv2DGroup_unit(srcptr_omp, kerptr, dstptr_omp, biasptr, rows,
				src_shape, dst_shape, wei_shape, group, stride_h, stride_w);
	}
#else
	if (is_depthwise)
		Conv2DDepthWise_unit(srcptr, kerptr, dstptr, biasptr, dst_shape[0],
			src_shape, dst_shape, wei_shape, stride_h, stride_w);
	else
		Conv2DGroup_unit(srcptr, kerptr, dstptr, biasptr, dst_shape[0],
			src_shape, dst_shape, wei_shape, group, stride_h, stride_w);
#endif

	out_ts = out_tensor;
}
