#include <assert.h>
#include <algorithm>

#include "matrix.h"
#include <immintrin.h>

using namespace std;

#define SIMD_WIDTH 8
#define SIMD 1

void outer (float *result, float *a, int dim_a, float *b, int dim_b) {
	if (!SIMD) {
		for (int i=0; i<dim_a; ++i) {
			for (int j=0; j<dim_b; ++j) {
				// R_ij = a_i * b_j
				result[i*dim_b+j] += a[i] * b[j];
			}
		}
	} else {
		cblas_sger(CblasRowMajor, dim_a, dim_b, 1.0, a, 1, b, 1, result, dim_b);
	}
}

void dot (float *result, float *A, int dim1_A, int dim2_A, float *B, int dim1_B, int dim2_B) {
	assert(dim2_A == dim1_B);
	if (!SIMD) {
		int dim_inner = dim2_A;
		for (int i=0; i<dim1_A; ++i) {
			for (int j=0; j<dim2_B; ++j) {
				for (int k=0; k<dim_inner; ++k) {
					// R_ij += A_ik * B_kj
					result[i*dim2_B+j] += A[i*dim2_A+k] * B[k*dim2_B+j];
				}
			}
		}
	} else {
		cblas_sgemv(CblasRowMajor, CblasNoTrans, dim1_A, dim2_A, 1.0, A, dim2_A, B, 1, 1.0, result, 1);
	}
}

void trans_dot (float *result, float *A, int dim1_A, int dim2_A, float *B, int dim1_B, int dim2_B) {
	assert(dim1_A == dim1_B);
	if (!SIMD) {
		int dim_inner = dim1_A;
		for (int i=0; i<dim2_A; ++i) {
			for (int j=0; j<dim2_B; ++j) {
				for (int k=0; k<dim_inner; ++k) {
					// R_ij += A_ki * B_ki
					result[i*dim2_B+j] += A[k*dim2_A+i] * B[k*dim2_B+j];
				}
			}
		}
	} else {
		cblas_sgemv(CblasRowMajor, CblasTrans, dim1_A, dim2_A, 1.0, A, dim2_A, B, 1, 1.0, result, 1);
	}
}

void elem_mul (float *result, float *a, float *b, int dim) {	
	if (!SIMD) {
		for (int i=0; i<dim; ++i) {
			// R_i += a_i * b_i
			result[i] += a[i] * b[i];
		}
	} else {
		__m256 vec_a, vec_b, vec_res;
		for (int i=0; i<dim; i+=SIMD_WIDTH) {
			vec_a = _mm256_loadu_ps(a + i);
			vec_b = _mm256_loadu_ps(b + i);
			vec_res = _mm256_loadu_ps(result + i);

			_mm256_mul_ps(vec_a, vec_b);
			_mm256_add_ps(vec_res, vec_a);
			_mm256_storeu_ps(result + i, vec_res);
		}
	}
}

void elem_mul_triple (float *result, float *a, float *b, float *c, int dim) {
	if (!SIMD) {
		for (int i=0; i<dim; ++i) {
			// R_i += a_i * b_i * c_i
			result[i] += a[i] * b[i] * c[i];
		}
	} else {
		__m256 vec_a, vec_b, vec_c, vec_res;
		for (int i=0; i<dim; i+=SIMD_WIDTH) {
			vec_a = _mm256_loadu_ps(a + i);
			vec_b = _mm256_loadu_ps(b + i);
			vec_c = _mm256_loadu_ps(c + i);
			vec_res = _mm256_loadu_ps(result + i);

			_mm256_mul_ps(vec_a, vec_b);
			_mm256_mul_ps(vec_a, vec_c);
			_mm256_add_ps(vec_res, vec_a);
			_mm256_storeu_ps(result + i, vec_res);
		}
	}
}

void elem_sub (float *result, float *a, float *b, int dim) {
	if (!SIMD) {
		for (int i=0; i<dim; ++i) {
			// R_i += a_i - b_i
			result[i] += a[i] - b[i];
		}
	} else {
		__m256 vec_a, vec_b, vec_res;
		for (int i=0; i<dim; i+=SIMD_WIDTH) {
			vec_a = _mm256_loadu_ps(a + i);
			vec_b = _mm256_loadu_ps(b + i);
			vec_res = _mm256_loadu_ps(result + i);

			_mm256_sub_ps(vec_a, vec_b);
			_mm256_add_ps(vec_res, vec_a);
			_mm256_storeu_ps(result + i, vec_res);
		}
	}
}

void elem_accum (float *result, float *a, int dim) {
	if (!SIMD) {
		for (int i=0; i<dim; ++i) {
			// R_i += a_i
			result[i] += a[i];
		}
	} else {
		__m256 vec_a, vec_res;
		for (int i=0; i<dim; i+=SIMD_WIDTH) {
			vec_a = _mm256_loadu_ps(a + i);
			vec_res = _mm256_loadu_ps(result + i);

			_mm256_add_ps(vec_res, vec_a);
			_mm256_storeu_ps(result + i, vec_res);
		}
	}
}