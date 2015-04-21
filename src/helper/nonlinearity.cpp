#include "nonlinearity.h"

/****************************************************************
* Single-thread version
****************************************************************/
void sigm (float *sigm_res, float *input, int dim) {
	if (!SIMD) {
		for (int i=0; i<dim; i++) {
			sigm_res[i] = 1 / (1 + exp(-input[i]));
		}
	} else {
		int residual = dim % SIMD_WIDTH;
		int stopSIMD = dim - residual;

		__m256 vec_input, vec_res;
		__m256 vec_zero = _mm256_set1_ps(0.f);
		__m256 vec_one  = _mm256_set1_ps(1.f);
		for (int i=0; i<stopSIMD; i+=SIMD_WIDTH) {
			vec_input = _mm256_loadu_ps(input + i);			
			// vec_res = _mm256_loadu_ps(sigm_res + i);

			vec_input = _mm256_exp_ps(_mm256_sub_ps(vec_zero, vec_input));
			vec_res = _mm256_div_ps(vec_one, _mm256_add_ps(vec_one, vec_input));
			_mm256_storeu_ps(sigm_res + i, vec_res);
		}

		for (int i=stopSIMD; i<dim; ++i) {
			sigm_res[i] = 1 / (1 + exp(-input[i]));
		}
	}
}

void sigm_deriv (float *deriv_res, float *sigm_res, int dim) {
	if (!SIMD) {
		for (int i=0; i<dim; i++) {
			deriv_res[i] = sigm_res[i] * (1 - sigm_res[i]);
		} 
	} else {
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
	}
}

void tanh (float *tanh_res, float *input, int dim) {
	if (!SIMD) {
		for (int i=0; i<dim; i++) {
			tanh_res[i] = tanh(input[i]);
		}
	} else {
		int residual = dim % SIMD_WIDTH;
		int stopSIMD = dim - residual;

		__m256 vec_input, vec_res;		
		for (int i=0; i<stopSIMD; i+=SIMD_WIDTH) {
			vec_input = _mm256_loadu_ps(input + i);
			// vec_res = _mm256_loadu_ps(tanh_res + i);
			
			vec_res = _mm256_tanh_ps(vec_input);
			_mm256_storeu_ps(tanh_res + i, vec_res);
		}

		for (int i=stopSIMD; i<dim; ++i) {
			tanh_res[i] = tanh(input[i]);
		}
	}
}

void tanh_deriv (float *deriv_res, float *tanh_res, int dim) {
	if (!SIMD) {
		for (int i=0; i<dim; i++) {
			deriv_res[i] = 1 - tanh_res[i] * tanh_res[i];
		}
	} else {
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