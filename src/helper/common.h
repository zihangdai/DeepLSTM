#ifndef __COMMON_HEADER_H__
#define __COMMON_HEADER_H__

#include <immintrin.h>

#define SIMD_WIDTH 8
#define SIMD 1
#define BLOCK_SIZE 64

_PS256_CONST(_ps256_exp_hi,	88.3762626647949f);
_PS256_CONST(_ps256_exp_lo,	-88.3762626647949f);

_PS256_CONST(_ps256_cephes_LOG2EF, 1.44269504088896341);
_PS256_CONST(_ps256_cephes_exp_C1, 0.693359375);
_PS256_CONST(_ps256_cephes_exp_C2, -2.12194440e-4);

_PS256_CONST(_ps256_cephes_exp_p0, 1.9875691500E-4);
_PS256_CONST(_ps256_cephes_exp_p1, 1.3981999507E-3);
_PS256_CONST(_ps256_cephes_exp_p2, 8.3334519073E-3);
_PS256_CONST(_ps256_cephes_exp_p3, 4.1665795894E-2);
_PS256_CONST(_ps256_cephes_exp_p4, 1.6666665459E-1);
_PS256_CONST(_ps256_cephes_exp_p5, 5.0000001201E-1);

__m256 exp256_ps(__m256 x);


#endif