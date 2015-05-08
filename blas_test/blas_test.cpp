#include <stdio.h>
#include <stdlib.h> 
#include <omp.h>
#include "cblas.h"
#include "cycle_timer.h"


int main(int argc, char const *argv[]) {
	if (argc < 4) {
		printf("Not enough arguments\n");
		return -1;
	}

	int max_num_thread = atoi(argv[1]);
	int max_iter = atoi(argv[2]);
	int test_method = atoi(argv[3]);

	openblas_set_num_threads(max_num_thread);
	omp_set_num_threads(max_num_thread);

	int m = 1024;
	int n = 1024;
	float *A = new float[m * n];
	for (int i = 0; i < m * n; i++) {
    	A[i] = rand() / RAND_MAX;
	}
	float *b = new float[n];
	for (int i = 0; i < n; i++) {
    	b[i] = rand() / RAND_MAX;
	}
	float *Ab = new float[m];

	switch (test_method) {
		case 0: {
			printf("Runing Matrix-Vector Multiplication by OpenMP (%d threads)\n", omp_get_max_threads());
			double begTime = CycleTimer::currentSeconds();
			for (int iter = 0; iter < max_iter; ++iter) {
			#pragma omp parallel for
				for (int i=0; i<m; ++i) {
					for (int j=0; j<n; ++j) {
						Ab[i] += A[i*n+j] * b[j];
					}
				}
			}
			double endTime = CycleTimer::currentSeconds();
			printf("%f\n", endTime - begTime);
			break;
		}
		case 1: {
			double begTime = CycleTimer::currentSeconds();
			printf("Runing Matrix-Vector Multiplication by OpenBlas (%d threads)\n", omp_get_max_threads());
			for (int iter = 0; iter < max_iter; ++iter) {
				cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, A, n, b, 1, 1.0, Ab, 1);
			}
			double endTime = CycleTimer::currentSeconds();
			printf("%f\n", endTime - begTime);
			break;
		}
		default:
			printf("No matched test method\n");
			break;
	}

	delete [] A;
	delete [] b;
	delete [] Ab;

	return 0;
}