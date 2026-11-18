#include "Activation.h"
#ifdef USE_optimization_OPENMP
#include <omp.h>
#endif

void antdnn::activation_leaky_relu(Tensor & ts, float alpha)
{
	int tsize = ts.size();
	auto ptrw = ts.ptr_write();

#ifdef USE_optimization_OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < tsize; i++)
		ptrw[i] = ptrw[i] >= 0 ? ptrw[i] : alpha * ptrw[i];
}
