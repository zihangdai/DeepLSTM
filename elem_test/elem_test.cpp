#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>
#include <algorithm>
#include "cycle_timer.h"

#define SIMD_WIDTH 8

void elem_mul (float *result, float *a, float *b, int dim) {
	#ifdef __linux
	int residual = dim % SIMD_WIDTH;
	int stopSIMD = dim - residual;

	__m256 vec_a, vec_b, vec_res;
	for (int i=0; i<stopSIMD; i+=SIMD_WIDTH) {
		vec_a = _mm256_loadu_ps(a + i);
		vec_b = _mm256_loadu_ps(b + i);
		vec_res = _mm256_loadu_ps(result + i);

		vec_a = _mm256_mul_ps(vec_a, vec_b);
		vec_res = _mm256_add_ps(vec_res, vec_a);
		_mm256_storeu_ps(result + i, vec_res);
	}

	for (int i=stopSIMD; i<dim; ++i) {
		result[i] += a[i] * b[i];
	}
	#endif
}

int main(int argc, char const *argv[]) {
	if (argc < 4) {
		printf("Not enough arguments\n");
		return -1;
	}

	int max_num_thread = atoi(argv[1]);
	int max_iter = atoi(argv[2]);
	int test_method = atoi(argv[3]);
	
	omp_set_num_threads(max_num_thread);

	int m = 1024;
	float *a0 = new float[m];
	for (int i = 0; i < m; i++) {
    	a0[i] = rand() / RAND_MAX;
	}
	float *b0 = new float[m];
	for (int i = 0; i < m; i++) {
    	b0[i] = rand() / RAND_MAX;
	}

	float *a1 = new float[m];
	for (int i = 0; i < m; i++) {
    	a1[i] = rand() / RAND_MAX;
	}
	float *b1 = new float[m];
	for (int i = 0; i < m; i++) {
    	b1[i] = rand() / RAND_MAX;
	}

	float *a2 = new float[m];
	for (int i = 0; i < m; i++) {
    	a2[i] = rand() / RAND_MAX;
	}
	float *b2 = new float[m];
	for (int i = 0; i < m; i++) {
    	b2[i] = rand() / RAND_MAX;
	}

	float *a3 = new float[m];
	for (int i = 0; i < m; i++) {
    	a3[i] = rand() / RAND_MAX;
	}
	float *b3 = new float[m];
	for (int i = 0; i < m; i++) {
    	b3[i] = rand() / RAND_MAX;
	}

	float *a4 = new float[m];
	for (int i = 0; i < m; i++) {
    	a4[i] = rand() / RAND_MAX;
	}
	float *b4 = new float[m];
	for (int i = 0; i < m; i++) {
    	b4[i] = rand() / RAND_MAX;
	}
	
	float *ab0 = new float[m];

	switch (test_method) {
		case 0: {
			printf("Runing Element-Wise Multiplication by OpenMP (%d threads)\n", omp_get_max_threads());
			double begTime = CycleTimer::currentSeconds();
			for (int iter = 0; iter < max_iter; ++iter) {
				#pragma omp parallel for
				for (int i=0; i<m; ++i) {					
					ab[i] += a0[i] * b0[i];
					ab[i] += a1[i] * b1[i];
					ab[i] += a2[i] * b2[i];
					ab[i] += a3[i] * b3[i];
					ab[i] += a4[i] * b4[i];
				}
			}
			double endTime = CycleTimer::currentSeconds();
			printf("%f\n", (endTime - begTime) / float(max_iter));
			break;
		}
		case 1: {
			double begTime = CycleTimer::currentSeconds();
			printf("Runing Matrix-Vector Multiplication by OpenBlas (%d threads)\n", omp_get_max_threads());
			for (int iter = 0; iter < max_iter; ++iter) {
				elem_mul(ab, a0, b0, m);
				elem_mul(ab, a1, b1, m);
				elem_mul(ab, a2, b2, m);
				elem_mul(ab, a3, b3, m);
				elem_mul(ab, a4, b4, m);
			}
			double endTime = CycleTimer::currentSeconds();
			printf("%f\n", (endTime - begTime) / float(max_iter));
			break;
		}
		case 2: {
			double begTime = CycleTimer::currentSeconds();
			printf("Runing Matrix-Vector Multiplication by OpenBlas (%d threads)\n", omp_get_max_threads());
			for (int iter = 0; iter < max_iter; ++iter) {
				#ifdef __linux
				int residual = dim % SIMD_WIDTH;
				int stopSIMD = dim - residual;

				__m256 vec_a0, vec_b0;
				__m256 vec_a1, vec_b1;
				__m256 vec_a2, vec_b2;
				__m256 vec_a3, vec_b3;
				__m256 vec_a4, vec_b4;
				__m256 vec_res;
				for (int i=0; i<stopSIMD; i+=SIMD_WIDTH) {
					vec_a0 = _mm256_loadu_ps(a0 + i);
					vec_b0 = _mm256_loadu_ps(b0 + i);
					
					vec_a1 = _mm256_loadu_ps(a1 + i);
					vec_b1 = _mm256_loadu_ps(b1 + i);
					
					vec_a2 = _mm256_loadu_ps(a2 + i);
					vec_b2 = _mm256_loadu_ps(b2 + i);
					
					vec_a3 = _mm256_loadu_ps(a3 + i);
					vec_b3 = _mm256_loadu_ps(b3 + i);
					
					vec_a4 = _mm256_loadu_ps(a4 + i);
					vec_b4 = _mm256_loadu_ps(b4 + i);

					vec_res = _mm256_loadu_ps(ab + i);

					vec_a0 = _mm256_mul_ps(vec_a0, vec_b0);
					vec_a1 = _mm256_mul_ps(vec_a1, vec_b1);
					vec_a2 = _mm256_mul_ps(vec_a2, vec_b2);
					vec_a3 = _mm256_mul_ps(vec_a3, vec_b3);
					vec_a4 = _mm256_mul_ps(vec_a4, vec_b4);

					vec_res = _mm256_add_ps(vec_res, vec_a0);
					vec_res = _mm256_add_ps(vec_res, vec_a1);
					vec_res = _mm256_add_ps(vec_res, vec_a2);
					vec_res = _mm256_add_ps(vec_res, vec_a3);
					vec_res = _mm256_add_ps(vec_res, vec_a4);

					_mm256_storeu_ps(ab + i, vec_res);
				}

				for (int i=stopSIMD; i<dim; ++i) {
					ab[i] += a0[i] * b0[i];
					ab[i] += a1[i] * b1[i];
					ab[i] += a2[i] * b2[i];
					ab[i] += a3[i] * b3[i];
					ab[i] += a4[i] * b4[i];
				}
				#endif
			}
			double endTime = CycleTimer::currentSeconds();
			printf("%f\n", (endTime - begTime) / float(max_iter));
			break;
		}
		case 3: {
			int block_size = (m + max_num_thread - 1)/ max_num_thread;
			double begTime = CycleTimer::currentSeconds();
			printf("Runing Matrix-Vector Multiplication by OpenMP (%d threads) with OpenBlas\n", omp_get_max_threads());
			for (int iter = 0; iter < max_iter; ++iter) {
				#pragma omp parallel for
				for (int i = 0; i < max_num_thread; ++i) {
					int start_idx = i*block_size;
					int actual_size = std::min(block_size, m-start_idx);					
					elem_mul(ab+start_idx, a0+start_idx, b0+start_idx, actual_size);
					elem_mul(ab+start_idx, a1+start_idx, b1+start_idx, actual_size);
					elem_mul(ab+start_idx, a2+start_idx, b2+start_idx, actual_size);
					elem_mul(ab+start_idx, a3+start_idx, b3+start_idx, actual_size);
					elem_mul(ab+start_idx, a4+start_idx, b4+start_idx, actual_size);
				}
			}
			double endTime = CycleTimer::currentSeconds();
			printf("%f\n", (endTime - begTime) / float(max_iter));
			break;
		}
		default:
			printf("No matched test method\n");
			break;
	}

	delete [] a0;
	delete [] b0;

	delete [] a1;
	delete [] b1;

	delete [] a2;
	delete [] b2;

	delete [] a3;
	delete [] b3;

	delete [] a4;
	delete [] b4;

	delete [] ab;

	return 0;
}