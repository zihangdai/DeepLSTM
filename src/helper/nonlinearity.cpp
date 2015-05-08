#include "nonlinearity.h"

/****************************************************************
* Single-thread version
****************************************************************/
void sigm (float *sigm_res, float *input, int dim) {	
	for (int i=0; i<dim; i++) {
		sigm_res[i] = 1 / (1 + exp(-input[i]));
	}
}

void sigm_deriv (float *deriv_res, float *sigm_res, int dim) {	
	if (!SIMD) {
		for (int i=0; i<dim; i++) {
			deriv_res[i] = sigm_res[i] * (1 - sigm_res[i]);
		} 
	} else {
		#ifdef __linux
		int residual = dim % SIMD_WIDTH;
		int stopSIMD = dim - residual;

		__m256 vec_deriv, vec_sigm;
		__m256 vec_one  = _mm256_set1_ps(1.f);
		for (int i=0; i<stopSIMD; i+=SIMD_WIDTH) {
			vec_sigm  = _mm256_loadu_ps(sigm_res + i);
			// vec_deriv = _mm256_loadu_ps(deriv_res + i);
			
			vec_deriv = _mm256_mul_ps(vec_sigm, _mm256_sub_ps(vec_one, vec_sigm));
			_mm256_storeu_ps(deriv_res + i, vec_deriv);
		}

		for (int i=stopSIMD; i<dim; ++i) {
			deriv_res[i] = sigm_res[i] * (1 - sigm_res[i]);
		}
		#endif
	}	
}

void tanh (float *tanh_res, float *input, int dim) {	
	for (int i=0; i<dim; i++) {
		tanh_res[i] = tanh(input[i]);
	}	
}

void tanh_deriv (float *deriv_res, float *tanh_res, int dim) {	
	if (!SIMD) {
		for (int i=0; i<dim; i++) {
			deriv_res[i] = 1 - tanh_res[i] * tanh_res[i];
		}
	} else {
		#ifdef __linux
		int residual = dim % SIMD_WIDTH;
		int stopSIMD = dim - residual;

		__m256 vec_deriv, vec_tanh;
		__m256 vec_one  = _mm256_set1_ps(1.f);
		for (int i=0; i<stopSIMD; i+=SIMD_WIDTH) {
			vec_tanh  = _mm256_loadu_ps(tanh_res + i);			
			// vec_deriv = _mm256_loadu_ps(deriv_res + i);
			
			vec_deriv = _mm256_sub_ps(vec_one, _mm256_mul_ps(vec_tanh, vec_tanh));
			_mm256_storeu_ps(deriv_res + i, vec_deriv);
		}

		for (int i=stopSIMD; i<dim; ++i) {
			deriv_res[i] = 1 - tanh_res[i] * tanh_res[i];
		}
		#endif
	}	
}

void softmax (float *result, float *input, int dim) {
	float maxInput = input[0];
	for (int i=1; i<dim; i++) {
		if (maxInput < input[i]) {
			maxInput = input[i];
		}
	}

	float sum = 0.f;
	for (int i=0; i<dim; i++) {
		result[i] = exp(input[i] - maxInput);
		sum += result[i];
	}
	for (int i=0; i<dim; i++) {
		result[i] /= sum;
	}
}

int argmax (float *input, int dim) {
	int maxIdx = 0;
	float maxVal = input[0];
	for (int idx=1; idx<dim; idx++) {
		if ( input[idx] > maxVal ) {
			maxIdx = idx;
			maxVal = input[idx];
		}
	}
	return maxIdx;
}

/****************************************************************
* Multi-thread version controled by OpenMP
****************************************************************/
void sigm_threads (float *sigm_res, float *input, int dim) {
	#pragma omp parallel for
	for (int i=0; i<dim; i++) {
		sigm_res[i] = 1 / (1 + exp(-input[i]));
	}
}

void sigm_deriv_threads (float *deriv_res, float *sigm_res, int dim) {
	#pragma omp parallel for
	for (int i=0; i<dim; i++) {
		deriv_res[i] = sigm_res[i] * (1 - sigm_res[i]);
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