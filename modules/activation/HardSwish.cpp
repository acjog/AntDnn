#include "Activation.h"
#ifdef USE_optimization_OPENMP
#include <omp.h>
#endif

void antdnn::activation_hardswish(Tensor & ts)
{
	int tsize = ts.size();
	auto ptrw = ts.ptr_write();

#ifdef USE_optimization_OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < tsize; i++)
	{
		float x = ptrw[i];
		// hardswish: x * relu6(x + 3) / 6
		float t = x + 3.0f;
		if (t > 6.0f)
			t = 6.0f;
		else if (t < 0.0f)
			t = 0.0f;
		ptrw[i] = x * t / 6.0f;
	}
}
