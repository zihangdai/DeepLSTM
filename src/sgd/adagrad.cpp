#include <math.h>
#include <string.h>
#include "sgd.h"

using namespace std;

adagrad::adagrad (ConfReader *confReader, int paramSize) {
	m_nParamSize = paramSize;
	m_learningRate = confReader->getFloat("learning_rate");
	m_useMomentum  = confReader->getInt("use_momentum");
	m_stepCount = 0;

	m_histSquareGrad = new float [m_nParamSize];
	for (int i=0; i<m_nParamSize; i++) {
		m_histSquareGrad[i] = 0.1f;
	}
}

adagrad::~adagrad () {
	if (!m_histSquareGrad) {
		delete [] m_histSquareGrad;
	}
}

void adagrad::updateParams (float *params, float *grad, int rank) {
	m_stepCount += 1;

	for (int i=0; i<m_nParamSize; i++) {
		m_histSquareGrad[i] += grad[i] * grad[i];
		params[i] -= m_learningRate * grad[i] / sqrt(m_histSquareGrad[i]);
	}	
}