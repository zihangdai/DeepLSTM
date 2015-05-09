#include <stdio.h>
#include <stdlib.h> 
#include <omp.h>
#include <algorithm>
#include "cycle_timer.h"

#define SIMD_WIDTH 8

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
	float *a = new float[m];
	for (int i = 0; i < m; i++) {
    	a[i] = rand() / RAND_MAX;
	}
	float *b = new float[m];
	for (int i = 0; i < m; i++) {
    	b[i] = rand() / RAND_MAX;
	}
	float *ab = new float[m];

	switch (test_method) {
		case 0: {
			printf("Runing Element-Wise Multiplication by OpenMP (%d threads)\n", omp_get_max_threads());
			double begTime = CycleTimer::currentSeconds();
			for (int iter = 0; iter < max_iter; ++iter) {
				#pragma omp parallel for
				for (int i=0; i<m; ++i) {					
					ab[i] += a[i] * b[i];
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
				#ifdef __linux
				int residual = m % SIMD_WIDTH;
				int stopSIMD = m - residual;

				__m256 vec_a, vec_b, vec_res;
				for (int i=0; i<stopSIMD; i+=SIMD_WIDTH) {
					vec_a = _mm256_loadu_ps(a + i);
					vec_b = _mm256_loadu_ps(b + i);
					vec_res = _mm256_loadu_ps(ab + i);

					vec_a = _mm256_mul_ps(vec_a, vec_b);
					vec_res = _mm256_add_ps(vec_res, vec_a);
					_mm256_storeu_ps(ab + i, vec_res);
				}

				for (int i=stopSIMD; i<m; ++i) {
					ab[i] += a[i] * b[i];
				}
				#endif
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
					#ifdef __linux
					int residual = actual_size % SIMD_WIDTH;
					int stopSIMD = actual_size - residual;

					__m256 vec_a, vec_b, vec_res;
					for (int i=start_idx; i<start_idx+stopSIMD; i+=SIMD_WIDTH) {
						vec_a = _mm256_loadu_ps(a + i);
						vec_b = _mm256_loadu_ps(b + i);
						vec_res = _mm256_loadu_ps(ab + i);

						vec_a = _mm256_mul_ps(vec_a, vec_b);
						vec_res = _mm256_add_ps(vec_res, vec_a);
						_mm256_storeu_ps(ab + i, vec_res);
					}

					for (int i=start_idx+stopSIMD; i<start_idx+actual_size; ++i) {
						ab[i] += a[i] * b[i];
					}
					#endif
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

	delete [] a;
	delete [] b;
	delete [] ab;

	return 0;
}