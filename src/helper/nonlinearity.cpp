#include "nonlinearity.h"

/****************************************************************
* Single-thread version
****************************************************************/
void sigm (float *simg_res, float *input, int dim) {
	for (int i=0; i<dim; i++) {
		simg_res[i] = 1 / (1 + exp(-input[i]));
	}
}

void sigm_deriv (float *deriv_res, float *simg_res, int dim) {
	for (int i=0; i<dim; i++) {
		deriv_res[i] = simg_res[i] * (1 - simg_res[i]);
	}
}

void tanh (float *tanh_res, float *input, int dim) {
	for (int i=0; i<dim; i++) {
		tanh_res[i] = tanh(input[i]);
	}
}

void tanh_deriv (float *deriv_res, float *tanh_res, int dim) {
	for (int i=0; i<dim; i++) {
		deriv_res[i] = 1 - tanh_res[i] * tanh_res[i];
	}
}

void softmax (float *result, float *input, int dim) {
	float sum = 0.f;
	for (int i=0; i<dim; i++) {
		result[i] = exp(input[i]);
		sum += result[i];
	}
	for (int i=0; i<dim; i++) {
		result[i] /= sum;
	}
}

/****************************************************************
* Multi-thread version controled by OpenMP
****************************************************************/
void sigm_threads (float *simg_res, float *input, int dim) {
	#pragma omp parallel for
	for (int i=0; i<dim; i++) {
		simg_res[i] = 1 / (1 + exp(-input[i]));
	}
}

void sigm_deriv_threads (float *deriv_res, float *simg_res, int dim) {
	#pragma omp parallel for
	for (int i=0; i<dim; i++) {
		deriv_res[i] = simg_res[i] * (1 - simg_res[i]);
	}
}

void tanh_threads (float *tanh_res, float *input, int dim) {
	#pragma omp parallel for
	for (int i=0; i<dim; i++) {
		tanh_res[i] = tanh(input[i]);
	}
}

void tanh_deriv_threads (float *deriv_res, float *tanh_res, int dim) {
	#pragma omp parallel for
	for (int i=0; i<dim; i++) {
		deriv_res[i] = 1 - tanh_res[i] * tanh_res[i];
	}
}

void softmax_threads (float *result, float *input, int dim) {
	float sum = 0.f;
	#pragma omp parallel for reduction(+:sum)
	for (int i=0; i<dim; i++) {
		result[i] = exp(input[i]);
		sum += result[i];
	}
	for (int i=0; i<dim; i++) {
		result[i] /= sum;
	}
}