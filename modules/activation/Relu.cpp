#include "Activation.h"
#include <math.h>
#ifdef USE_optimization_OPENMP
#include <omp.h>
#endif
#define __max(a,b)  (((a) > (b)) ? (a) : (b))
void antdnn::activation_relu(Tensor & ts)
{
	int tsize = ts.size();
	auto ptrw = ts.ptr_write();

#if USE_optimization_AVX==1
	if (tsize < 8)
	{
#ifdef USE_optimization_OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < tsize; i++)
			ptrw[i] = __max(ptrw[i], 0);
	}
	else
	{
		size_t cntBlock = tsize / 8;    // blocks
		size_t cntRem = tsize % 8;
		__m256 avx_zeros = _mm256_setzero_ps();
#ifdef USE_optimization_OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < cntBlock; i++)
		{
			auto tmpptr = ptrw + i * 8;
			*(__m256*)(tmpptr) = _mm256_max_ps(avx_zeros, *(__m256*)tmpptr);
		}
		for (int i = tsize - cntRem; i < tsize; i++)
		{
			ptrw[i] = __max(ptrw[i], 0);
		}
	}
		
#else
#ifdef USE_optimization_OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < tsize; i++)
		ptrw[i] = __max(ptrw[i], 0);
#endif
}
