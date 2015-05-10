#include <stdio.h>
#include <stdlib.h> 
#include <omp.h>
#include <algorithm>
#include <immintrin.h>
#include "cblas.h"
#include "cycle_timer.h"

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
	if (argc < 2) {
		printf("Not enough arguments\n");
		return -1;
	}

	int test_method = atoi(argv[1]);

	openblas_set_num_threads(1);	

	int m = 1024;
	int n = 1024;
	float *A = new float[m * n];
	for (int i = 0; i < m * n; i++) {
    	A[i] = rand() / RAND_MAX;
	}
	float *a = new float[n];
	for (int i = 0; i < n; i++) {
    	a[i] = rand() / RAND_MAX;
	}

	float *B = new float[m * n];
	for (int i = 0; i < m * n; i++) {
    	B[i] = rand() / RAND_MAX;
	}	
	float *b = new float[n];
	for (int i = 0; i < n; i++) {
    	b[i] = rand() / RAND_MAX;
	}

	float *C = new float[m];
	float *c = new float[m];

	float *temp_a = new float[m];
	float *temp_b = new float[m];

	float *res = new float[m];

	switch (test_method) {
		case 0: {
			omp_set_num_threads(3);
			double begTime = CycleTimer::currentSeconds();			
			#pragma omp parallel for
			for (int i=0; i<3; ++i) {
				switch(i) {
					case 0: {
						cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, A, n, a, 1, 1.0, temp_a, 1);
						break;
					}
					case 1: {
						cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, B, n, b, 1, 1.0, temp_b, 1);
						break;
					}
					case 2: {
						elem_mul(res, C, c, m);
						break;
					}					
				}
			}
			for (int i=0; i<m; ++i) {
				res[i] += temp_a[i] + temp_b[i];
			}
			double endTime = CycleTimer::currentSeconds();
			printf("%f\n", (endTime - begTime));
			break;
		}
		case 1: {
			double begTime = CycleTimer::currentSeconds();			
			cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, A, n, a, 1, 1.0, res, 1);
			cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, B, n, b, 1, 1.0, res, 1);
			elem_mul(res, C, c, m);
			double endTime = CycleTimer::currentSeconds();
			printf("%f\n", (endTime - begTime));
			break;
		}		
		default: {
			printf("No matched test method\n");
			return -1;
		}
	}

	delete [] A;
	delete [] B;
	delete [] a;
	delete [] b;
	delete [] C;
	delete [] c;

	delete [] temp_a;
	delete [] temp_b;
	delete [] res;

	return 0;
}