#ifndef __COMMON_HEADER_H__
#define __COMMON_HEADER_H__

#include <immintrin.h>

#define SIMD_WIDTH 8
#define SIMD 1
#define BLOCK_SIZE 64

__m256 exp256_ps(__m256 x);

#endif