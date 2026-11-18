#pragma once
// OpenMP: only enable when compiling with a toolchain that actually provides
// omp.h (MSVC, GCC/MinGW, or Clang with libomp). Apple Clang on macOS has no
// OpenMP support by default, so leave it off there.
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
#define USE_optimization_OPENMP
#endif
#define USE_optimization_AVX 0
#define OpenRunTimeDetect 1


#ifdef _MSC_VER
	#ifdef _DLL_ANTDNN
		#define DLL_ANTDNN_API __declspec(dllexport)
	#else  
		#define DLL_ANTDNN_API __declspec(dllimport)
	#endif  
#else
	#define DLL_ANTDNN_API
#endif // _MSC_VER

