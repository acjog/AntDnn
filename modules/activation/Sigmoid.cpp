#include "Activation.h"
#include <math.h>
#ifdef USE_optimization_OPENMP
#include <omp.h>
#endif

void antdnn::activation_sigmoid(Tensor & ts)
{
	int tsize = ts.size();
	auto ptrw = ts.ptr_write();

#ifdef USE_optimization_OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < tsize; i++)
		ptrw[i] = 1.0f / (1.0f + expf(-ptrw[i]));
}
