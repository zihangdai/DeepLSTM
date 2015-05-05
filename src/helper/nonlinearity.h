#ifndef __HELPER_NONLINEARITY_H__
#define __HELPER_NONLINEARITY_H__

#include <math.h>
#include "common.h"

/****************************************************************
* NOTE: It's not always good to use the multi-thread version of 
* the library because each computation is so few that the cost 
* of spawning threads may even be higher. Thus, the multi-thread
* functions may be preferred only when the dimension is extremely
* high. 
****************************************************************/

/****************************************************************
* Single-thread version
****************************************************************/
void sigm (float *result, float *input, int dim);

void sigm_deriv (float *result, float *input, int dim);

void tanh (float *result, float *input, int dim);

void tanh_deriv (float *result, float *input, int dim);

void softmax (float *result, float *input, int dim);

int argmax (float *input, int dim);

/****************************************************************
* Multi-thread version controled by OpenMP
****************************************************************/

void sigm_threads (float *result, float *input, int dim);

void sigm_deriv_threads (float *result, float *input, int dim);

void tanh_threads (float *result, float *input, int dim);

void tanh_deriv_threads (float *result, float *input, int dim);

void softmax_threads (float *result, float *input, int dim);

#endif