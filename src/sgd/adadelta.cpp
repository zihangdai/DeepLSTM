#include <math.h>
#include <string.h>
#include "sgd.h"

adadelta::adadelta (ConfReader *confReader, int paramSize) : sgdBase(confReader, paramSize){
	m_decayFactor = confReader->getFloat("adadelta_decay_factor");
	m_stableConst = confReader->getFloat("adadelta_stable_const");		

	m_ESquareGrad  = new float [m_nParamSize];
	m_ESquareDelta = new float [m_nParamSize];

	memset(m_ESquareGrad, 0x00, sizeof(float) * m_nParamSize);
	memset(m_ESquareDelta, 0x00, sizeof(float) * m_nParamSize);
}

adadelta::~adadelta () {
	if (m_ESquareGrad != NULL) {
		delete [] m_ESquareGrad;
	}
	if (m_ESquareDelta != NULL) {
		delete [] m_ESquareDelta;
	}
}

void adadelta::updateParams (float *params, float *grad, int rank) {
	float delta;
	for (int i=0; i<m_nParamSize; i++) {
		// accumulate mean squared grad
		m_ESquareGrad[i] = m_decayFactor * m_ESquareGrad[i] + (1 - m_decayFactor) * grad[i] * grad[i];
		// compute delta
		delta = sqrt(m_ESquareDelta[i] + m_stableConst) / sqrt(m_ESquareGrad[i] + m_stableConst) * grad[i];
		m_velocity[i] = m_momentumFactor * m_velocity[i] - delta;
		params[i] += m_velocity[i];
		// accumulate mean squared delta
		m_ESquareDelta[i] = m_decayFactor * m_ESquareDelta[i] + (1 - m_decayFactor) * delta * delta;
	}
}