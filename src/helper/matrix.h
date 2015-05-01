#ifndef __HELPER_MATRIX_H__
#define __HELPER_MATRIX_H__


#include <math.h>
#include <algorithm>
#include "cblas.h"
#include "common.h"

void outer (float *result, float *a, int dim_a, float *b, int dim_b);

void dot (float *result, float *A, int dim1_A, int dim2_A, float *B, int dim1_B, int dim2_B);

void trans_dot (float *result, float *A, int dim1_A, int dim2_A, float *B, int dim1_B, int dim2_B);

void elem_mul (float *result, float *a, float *b, int dim);

void elem_mul_triple (float *result, float *a, float *b, float *c, int dim);

void elem_sub (float *result, float *a, float *b, int dim);

void elem_accum (float *result, float *a, int dim);

void dot_threads (float *result, float *A, int dim1_A, int dim2_A, float *B, int dim1_B, int dim2_B);

void elem_mul_threads (float *result, float *a, float *b, int dim);

#endif