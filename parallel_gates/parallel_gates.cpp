#include <stdio.h>
#include <stdlib.h> 
#include <omp.h>
#include <algorithm>
#include <immintrin.h>
#include "cblas.h"
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
	if (argc < 2) {
		printf("Not enough arguments\n");
		return -1;
	}

	int test_method = atoi(argv[1]);

	openblas_set_num_threads(1);	

	int m = 1024;
	int n = 1024;
	float *A_i = new float[m * n];
	for (int i = 0; i < m * n; i++) {
    	A_i[i] = rand() / RAND_MAX;
	}
	float *A_f = new float[m * n];
	for (int i = 0; i < m * n; i++) {
    	A_f[i] = rand() / RAND_MAX;
	}
	float *A_s = new float[m * n];
	for (int i = 0; i < m * n; i++) {
    	A_s[i] = rand() / RAND_MAX;
	}
	float *a = new float[n];
	for (int i = 0; i < n; i++) {
    	a[i] = rand() / RAND_MAX;
	}

	float *B_i = new float[m * n];
	for (int i = 0; i < m * n; i++) {
    	B_i[i] = rand() / RAND_MAX;
	}
	float *B_f = new float[m * n];
	for (int i = 0; i < m * n; i++) {
    	B_f[i] = rand() / RAND_MAX;
	}
	float *B_s = new float[m * n];
	for (int i = 0; i < m * n; i++) {
    	B_s[i] = rand() / RAND_MAX;
	}
	float *b = new float[n];
	for (int i = 0; i < n; i++) {
    	b[i] = rand() / RAND_MAX;
	}

	float *C_i = new float[m];
	float *C_f = new float[m];
	float *c = new float[m];

	float *temp_a = new float[m];
	float *temp_b = new float[m];

	float *gate_i = new float[m];
	float *gate_f = new float[m];
	float *gate_s = new float[m];

	switch (test_method) {
		case 0: {
			omp_set_num_threads(3);
			double begTime = CycleTimer::currentSeconds();			
			#pragma omp parallel for
			for (int i=0; i<3; ++i) {
				switch(i) {
					case 0: {
						cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, A_i, n, a, 1, 1.0, gate_i, 1);
						cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, B_i, n, b, 1, 1.0, gate_i, 1);
						elem_mul(gate_i, C_i, c, m);
						break;
					}
					case 1: {
						cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, A_f, n, a, 1, 1.0, gate_f, 1);
						cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, B_f, n, b, 1, 1.0, gate_f, 1);
						elem_mul(gate_f, C_f, c, m);
						break;
					}
					case 2: {
						cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, A_s, n, a, 1, 1.0, gate_s, 1);
						cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, B_s, n, a, 1, 1.0, gate_s, 1);
						break;
					}					
				}
			}			
			double endTime = CycleTimer::currentSeconds();
			printf("%f\n", (endTime - begTime));
			break;
		}
		case 1: {
			double begTime = CycleTimer::currentSeconds();
			// cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, A_i, n, a, 1, 1.0, gate_i, 1);
			cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, B_i, n, b, 1, 1.0, gate_i, 1);
			elem_mul(gate_i, C_i, c, m);
			// cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, A_f, n, a, 1, 1.0, gate_f, 1);
			cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, B_f, n, b, 1, 1.0, gate_f, 1);
			elem_mul(gate_f, C_f, c, m);
			// cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, A_s, n, a, 1, 1.0, gate_s, 1);
			cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, B_s, n, a, 1, 1.0, gate_s, 1);
			double endTime = CycleTimer::currentSeconds();
			printf("%f\n", (endTime - begTime));
			break;
		}		
		default: {
			printf("No matched test method\n");
			return -1;
		}
	}

	delete [] A_i;
	delete [] B_i;
	delete [] C_i;

	delete [] A_f;
	delete [] B_f;
	delete [] C_f;

	delete [] A_s;
	delete [] B_s;
	
	delete [] a;
	delete [] b;	
	delete [] c;

	delete [] gate_i;
	delete [] gate_f;
	delete [] gate_s;

	return 0;
}