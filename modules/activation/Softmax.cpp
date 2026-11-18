#include "Activation.h"
#include <math.h>
#ifdef USE_optimization_OPENMP
#include <omp.h>
#endif
#define __max(a,b)  (((a) > (b)) ? (a) : (b))

void antdnn::activation_softmax(Tensor & ts)
{
	auto ptrw = ts.ptr_write();
	const int size_channels = ts.shape()[ts.dim() - 1];
	int size_pixel = ts.size() / size_channels;
	float sum, maxval;
	for (int i = 0; i < size_pixel; i++)
	{
		// 数值稳定: 先减去最大值, 避免 exp 溢出产生 inf/nan
		maxval = ptrw[0];
		for (int c = 1; c < size_channels; c++)
		{
			if (ptrw[c] > maxval)
				maxval = ptrw[c];
		}
		sum = 0;
		for (int c = 0; c < size_channels; c++)
		{
			ptrw[c] = expf(ptrw[c] - maxval);
			sum += ptrw[c];
		}
		for (int c = 0; c < size_channels; c++)
		{
			ptrw[c] /= sum;
		}
		ptrw += size_channels;
	}
	return void();
}
