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
	float *ab0 = new float[m];

	float *a1 = new float[m];
	for (int i = 0; i < m; i++) {
    	a1[i] = rand() / RAND_MAX;
	}
	float *b1 = new float[m];
	for (int i = 0; i < m; i++) {
    	b1[i] = rand() / RAND_MAX;
	}
	float *ab1 = new float[m];

	float *a2 = new float[m];
	for (int i = 0; i < m; i++) {
    	a2[i] = rand() / RAND_MAX;
	}
	float *b2 = new float[m];
	for (int i = 0; i < m; i++) {
    	b2[i] = rand() / RAND_MAX;
	}
	float *ab2 = new float[m];

	float *a3 = new float[m];
	for (int i = 0; i < m; i++) {
    	a3[i] = rand() / RAND_MAX;
	}
	float *b3 = new float[m];
	for (int i = 0; i < m; i++) {
    	b3[i] = rand() / RAND_MAX;
	}
	float *ab3 = new float[m];

	switch (test_method) {
		case 0: {
			printf("Runing Element-Wise Multiplication by OpenMP (%d threads)\n", omp_get_max_threads());
			double begTime = CycleTimer::currentSeconds();
			for (int iter = 0; iter < max_iter; ++iter) {
				#pragma omp parallel for
				for (int i=0; i<m; ++i) {					
					ab0[i] += a0[i] * b0[i];
					ab1[i] += a1[i] * b1[i];
					ab2[i] += a2[i] * b2[i];
					ab3[i] += a3[i] * b3[i];
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
				elem_mul(ab0, a0, b0, m);
				elem_mul(ab1, a1, b1, m);
				elem_mul(ab2, a2, b2, m);
				elem_mul(ab3, a3, b3, m);
			}
			double endTime = CycleTimer::currentSeconds();
			printf("%f\n", (endTime - begTime) / float(max_iter));
			break;
		}
		case 2: {
			int block_size = (m + max_num_thread - 1)/ max_num_thread;			
			double begTime = CycleTimer::currentSeconds();
			printf("Runing Matrix-Vector Multiplication by OpenMP (%d threads) with OpenBlas\n", omp_get_max_threads());
			for (int iter = 0; iter < max_iter; ++iter) {
				#pragma omp parallel for
				for (int i = 0; i < max_num_thread; ++i) {
					int start_idx = i*block_size;
					int actual_size = std::min(block_size, m-start_idx);					
					elem_mul(ab0+start_idx, a0+start_idx, b0+start_idx, actual_size);
					elem_mul(ab1+start_idx, a1+start_idx, b1+start_idx, actual_size);
					elem_mul(ab2+start_idx, a2+start_idx, b2+start_idx, actual_size);
					elem_mul(ab3+start_idx, a3+start_idx, b3+start_idx, actual_size);
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
	delete [] ab0;

	delete [] a1;
	delete [] b1;
	delete [] ab1;

	delete [] a2;
	delete [] b2;
	delete [] ab2;

	delete [] a3;
	delete [] b3;
	delete [] ab3;

	return 0;
}