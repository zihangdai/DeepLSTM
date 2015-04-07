#include <assert.h>
#include "matrix.h"

void outer (float *result, float *a, int dim_a, float *b, int dim_b) {
	for (int i=0; i<dim_a; ++i) {
		for (int j=0; j<dim_b; ++j) {
			// R_ij = a_i * b_j
			result[i*dim_b+j] += a[i] * b[j];
		}
	}
}

void dot (float *result, float *A, int dim1_A, int dim2_A, float *B, int dim1_B, int dim2_B) {
	assert(dim2_A == dim1_B);
	int dim_inner = dim2_A;
	for (int i=0; i<dim1_A; ++i) {
		for (int j=0; j<dim2_B; ++j) {
			for (int k=0; k<dim_inner; ++k) {
				// R_ij += A_ik * B_ki
				result[i*dim2_B+j] += A[i*dim_inner+k] * B[k*dim2_B+j];
			}
		}
	}
}

void trans_dot (float *result, float *A, int dim1_A, int dim2_A, float *B, int dim1_B, int dim2_B) {
	assert(dim1_A == dim1_B);
	int dim_inner = dim1_A;
	for (int i=0; i<dim2_A; ++i) {
		for (int j=0; j<dim2_B; ++j) {
			for (int k=0; k<dim_inner; ++k) {
				// R_ij += A_ki * B_ki
				result[i*dim2_B+j] += A[k*dim2_A+i] * B[k*dim2_B+j];
			}
		}
	}
}

void elem_mul (float *result, float *a, float *b, int dim) {
	for (int i=0; i<dim; ++i) {
		// R_i += a_i * b_i
		result[i] += a[i] * b[i];
	}
}

void elem_mul_triple (float *result, float *a, float *b, float *c, int dim) {
	for (int i=0; i<dim; ++i) {
		// R_i += a_i * b_i * c_i
		result[i] += a[i] * b[i] * c[i];
	}
}

void elem_sub (float *result, float *a, float *b, int dim) {
	for (int i=0; i<dim; ++i) {
		// R_i += a_i - b_i
		result[i] += a[i] - b[i];
	}	
}

void elem_accum (float *result, float *a, int dim) {
	for (int i=0; i<dim; ++i) {
		// R_i += a_i
		result[i] += a[i];
	}	
}