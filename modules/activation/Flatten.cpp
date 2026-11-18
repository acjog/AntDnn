#include "Activation.h"
#include <math.h>
#ifdef USE_optimization_OPENMP
#include <omp.h>
#endif

void antdnn::flatten(Tensor & ts)
{
	int size = ts.size();
	ts.reshape(1, &size);
	return void();
}
