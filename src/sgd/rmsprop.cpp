#include <math.h>
#include <string.h>
#include "sgd.h"

rmsprop::rmsprop (ConfReader *confReader, int paramSize) {
	m_nParamSize = paramSize;
	m_decayFactor = confReader->getFloat("rmsprop decay factor");
	m_useMomentum  = confReader->getInt("use momentum");

	m_meanSquareGrad  = new float [m_nParamSize];

	memset(m_meanSquareGrad, 0x00, sizeof(float) * m_nParamSize);
}

rmsprop::~rmsprop () {
	if (!m_meanSquareGrad) {
		delete [] m_meanSquareGrad;
	}
}

void rmsprop::updateParams (float *params, float *grad, int rank) {
	float delta;
	for (int i=0; i<m_nParamSize; i++) {
		// accumulate mean squared grad
		m_meanSquareGrad[i] = m_decayFactor * m_meanSquareGrad[i] + (1 - m_decayFactor) * grad[i] * grad[i];
		// compute delta
		params[i] -= grad[i] / sqrt(m_meanSquareGrad[i]);
	}
}