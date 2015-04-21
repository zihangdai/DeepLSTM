#include <math.h>
#include <string.h>
#include <omp.h>
#include "sgd.h"
#include "common.h"

using namespace std;

adagrad::adagrad (ConfReader *confReader, int paramSize) : sgdBase(confReader, paramSize) {
	m_learningRate = confReader->getFloat("learning_rate");		

	m_histSquareGrad = new float [m_nParamSize];
	for (int i=0; i<m_nParamSize; i++) {
		m_histSquareGrad[i] = 0.1f;
	}
}

adagrad::~adagrad () {
	if (m_histSquareGrad != NULL) {
		delete [] m_histSquareGrad;
	}
}

void adagrad::updateParams (float *params, float *grad, int rank) {
	m_stepCount += 1;

	elem_mul(m_histSquareGrad, grad, grad, m_nParamSize);	
	for (int i=0; i<m_nParamSize; i++) {
		// m_histSquareGrad[i] += grad[i] * grad[i];
		m_velocity[i] = m_momentumFactor * m_velocity[i] - m_learningRate * grad[i] / sqrt(m_histSquareGrad[i]);
		// params[i] += m_velocity[i];
	}
	elem_accum(params, m_velocity, m_nParamSize);
}