#ifndef __HELPER_NONLINEARITY_H__
#define __HELPER_NONLINEARITY_H__

#include <math.h>

void sigm (float *result, float *input, int dim);

void sigm_deriv (float *result, float *input, int dim);

void tanh (float *result, float *input, int dim);

void tanh_deriv (float *result, float *input, int dim);

void softmax (float *result, float *input, int dim);

#endif